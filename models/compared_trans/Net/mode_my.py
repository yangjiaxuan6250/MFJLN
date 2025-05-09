
from UDL.Basis.module import PatchMergeModule
from models.base_model import DerainModel
from torch import optim
from thop import profile
from UDL.Basis.criterion_metrics import SetCriterion
from .losses import CharbonnierLoss, EdgeLoss, fftLoss



import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers

from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class FeedForward(nn.Module):
    def __init__(self, patch_size, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.conv1 = Partial_conv3(dim)
        self.conv2 = nn.Conv2d(dim, hidden_features *2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x1, x2 = self.conv2(x).chunk(2, dim=1)

        x = self.project_out(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.proj1 = Partial_conv3(cnn_in)
        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.proj1(cx)
        cx = self.conv1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)
        return hx



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, patch_size, thres=0.1, suppress=0.75, pool_size=2, cut_num=4, cut_low=2):
        super(Attention, self).__init__()
        self.__class__.__name__ = 'XCTEB'
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.max_relative_position = 2
        per_dim = dim // cut_num
        self.atten_dim = atten_dim = cut_low * per_dim
        self.high_dim = high_dim = (cut_num - cut_low) * per_dim
        self.high_mixer = HighMixer(high_dim)
        self.conv_fuse = nn.Conv2d(atten_dim + high_dim * 2, atten_dim + high_dim * 2, kernel_size=3, stride=1,
                                   padding=1,
                                   bias=False, groups=atten_dim + high_dim * 2)
        self.pool_size = pool_size
        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv = Partial_conv3(atten_dim)
        self.qkv_conv = nn.Conv2d(atten_dim, atten_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(atten_dim+high_dim*2, dim, kernel_size=1, stride=1, padding=0)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()
        self.time_weighting = nn.Parameter(torch.ones(self.num_heads, atten_dim // self.num_heads, atten_dim // self.num_heads))
        # for suppressing
        self.thres = thres
        self.suppress_status = True if suppress != 0 else False
        if self.suppress_status:
            self.suppress = nn.Parameter(torch.tensor(suppress))  # suppress factor
    def forward(self, x):
        # q, k, v = self.qkv(x, chunk=3)
        # qkv = self.qkv_cheap(torch.cat([q, k, v], dim=1))
        b, c, h, w = x.shape
        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.pool(lx)

        qkv = self.qkv_conv(self.qkv(lx))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) * self.time_weighting


        lx = (attn @ v)

        lx = rearrange(lx, 'b head c (h w) -> b (head c) h w', head=self.num_heads,h=h//self.pool_size, w=w//self.pool_size)
        lx = self.uppool(lx)
        out = torch.cat((hx, lx), dim=1)
        out = out + self.conv_fuse(out)
        out = self.project_out(out)

        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, patch_size=1, cut_num=4, cut_low=2):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, patch_size, cut_num=cut_num, cut_low=cut_low)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(patch_size, dim, ffn_expansion_factor, bias)

        self.main_fft = nn.Sequential(
            BasicConv(dim * 2, dim * 2, kernel_size=1, stride=1, relu=False),
            nn.GELU(),
            BasicConv(dim * 2, dim * 2, kernel_size=1, stride=1, relu=False)
        )
        self.norm = 'backward'
    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(h, w), norm=self.norm)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = x + y


        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Together_3(nn.Module):
    def __init__(self, dim, reduction=16):
        super(Together_3, self).__init__()
        self.down2 = Downsample(dim // 2)
        self.down1_1 = Downsample(dim // 4)
        self.down1_2 = Downsample(dim // 2)
        self.conv = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, e1, e2, e3, d):
        e1 = self.down1_2(self.down1_1(e1))
        e2 = self.down2(e2)
        x = torch.cat([e1, e2, e3, d], dim=1)
        x = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Together_2(nn.Module):
    def __init__(self, dim, reduction=16):
        super(Together_2, self).__init__()
        self.down1 = Downsample(dim // 2)
        self.up3 = Upsample(dim * 2)
        self.conv = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, e1, e2, e3, d):
        e1 = self.down1(e1)
        e3 = self.up3(e3)
        x = torch.cat([e1, e2, e3, d], dim=1)
        x = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Together_1(nn.Module):
    def __init__(self, dim, reduction=16):
        super(Together_1, self).__init__()
        self.up2 = Upsample(dim * 2)
        self.up3_1 = Upsample(dim * 4)
        self.up3_2 = Upsample(dim * 2)
        self.conv = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, e1, e2, e3, d):
        e2 = self.up2(e2)
        e3 = self.up3_2(self.up3_1(e3))
        x = torch.cat([e1, e2, e3, d], dim=1)
        x = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################
##---------- Restormer -----------------------
class Net(PatchMergeModule):
    def __init__(self,
                 args,
                 patch_size=[64, 32, 16, 8],
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3, 4],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 n_embedding=64,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 cut_nums=[8, 8, 8, 8],
                 cut_low=[3, 4, 5, 6],
                 ):

        super(Net, self).__init__()
        self.args = args
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.prompt1 = PromptGenBlock(prompt_dim=32, prompt_len=5, prompt_size=64, lin_dim=96)
        # self.prompt2 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=32, lin_dim=192)
        # self.prompt3 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=16, lin_dim=384)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0], cut_low=cut_low[0]) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1], cut_low=cut_low[1]) for i in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2]) for i in
            range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[3], dim=int(dim * 2 ** 3), num_heads=heads[3],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[3], cut_low=cut_low[3]) for i in
            range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.together3 = Together_3(int(dim * 2 ** 2))

        # self.noise_level3 = TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 3) + 128, num_heads=heads[2],
        #                      ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2])
        # self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 3) + 128, int(dim * 2 ** 3), kernel_size=1, bias=bias)

        # self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2]) for i in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.together2 = Together_2(int(dim * 2 ** 1))

        # self.noise_level2 = TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 2) + 64, num_heads=heads[1],
        #                                      ffn_expansion_factor=ffn_expansion_factor,
        #                                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1],
        #                                      cut_low=cut_low[1])
        # self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 2) + 64, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1], cut_low=cut_low[1]) for i in
            range(num_blocks[1])])

        # self.noise_level1 = TransformerBlock(patch_size=patch_size[0], dim=int(dim * 2 ** 1) + 32, num_heads=heads[0],
        #                                      ffn_expansion_factor=ffn_expansion_factor,
        #                                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0],
        #                                      cut_low=cut_low[0])
        # self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 32, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.together1 = Together_1(dim)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0], cut_low=cut_low[0]) for i in
            range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # res = self.res(out_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        # latent += res
        # dec3_param = self.prompt3(latent)
        # latent = torch.cat([latent, dec3_param], dim=1)
        # latent = self.noise_level3(latent)
        # latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = self.together3(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # dec2_param = self.prompt2(out_dec_level3)
        # out_dec_level3 = torch.cat([out_dec_level3, dec2_param], dim=1)
        # out_dec_level3 = self.noise_level2(out_dec_level3)
        # out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self.together2(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # dec1_param = self.prompt1(out_dec_level2)
        # out_dec_level2 = torch.cat([out_dec_level2, dec1_param], dim=1)
        # out_dec_level2 = self.noise_level1(out_dec_level2)
        # out_dec_level2 = self.reduce_noise_level1(out_dec_level2)


        inp_dec_level1 = self.up2_1(out_dec_level2)
        # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.together1(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return [out_dec_level1]



    def train_step(self, *args, **kwargs):

        return self(*args, **kwargs)

    def val_step(self, *args, **kwargs):

        return self.forward_chop(*args, **kwargs)







class build_Net(DerainModel, name='Net'):
    def __call__(self, args):
        scheduler = dict(type='CosineAnnealingLrUpdaterHook', policy='CosineAnnealing', min_lr=1e-6,by_epoch=True)
        # lrs = lr_scheduler(0.1, 500)
        # scheduler = None
        l1_loss = CharbonnierLoss()
        edge_loss = EdgeLoss()
        f_loss = fftLoss()
        # S3imLoss = S3IM().cuda()
        # loss = BCMSLoss().cuda()
        weight_dict = {'loss': 1}
        losses = {'l1_loss': l1_loss, 'edge_loss': edge_loss, 'f_loss': f_loss}
        criterion = SetCriterion(losses, weight_dict)
        model = Net(args).cuda()
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)  ## optimizer 1: Adam
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        flops, params = profile(model, inputs=(dummy_input,))
        print('FLOPs: %.2f M, Params: %.2f M' % (flops / 1e6, params / 1e6))
        return model, criterion, optimizer, scheduler
