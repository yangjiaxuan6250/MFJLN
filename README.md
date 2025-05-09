
<div align="center">

# MFJLN: Multi-Frequency Feature Joint Learning Network for Rain Removal

</div>

## 🔥Training and Testing

### Training
**Step1.**
* Download datasets and put it with the following format. 
<table>
  <tr>
    <th align="left">Derain</th>
    <th align="center">Dataset</th>
  </tr>
  <tr>
    <td align="left">Rain200L</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
  </tr>
  <tr>
    <td align="left">Rain200H</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
  </tr>
  <tr>
    <td>DID</td>
    <td align="center"><a href="https://github.com/hezhangsprinter/DID-MDN">Link</a></td>
  </tr>
  <tr>
    <td>DDN</td>
    <td align="center"><a href="https://xueyangfu.github.io/projects/cvpr2017.html">Link</a></td>
  </tr>
<tr>
    <td>SPA</td>
    <td align="center"><a href="https://github.com/stevewongv/SPANet">Link</a></td>
  </tr>
</table>

* Verify the dataset path in `configs/configs.py`.
```
|-$ROOT/data
├── Rain200H
│   ├── train_c
│   │   ├── norain-1.png
│   │   ├── ...
│   ├── test_c
│   │   │   ├── norain-1.png
│   │   │   ├── ...
```

**Step2.** 
Open codes in your ide,  run the following code:

```
python run_derain.py
```

* A training example：

>	run_derain.py
  
	where arch='Restormer', and configs/option_Restormer.py has: 
  
	__cfg.eval__ = False, 
  
	__cfg.workflow__ = [('train', 50)], __cfg.dataset__ = {'train': 'Rain200H'}
	
* A test example:

>	run_derain_test.py

  	__cfg.dataset__ = {'val': 'Rain200H'}

	__cfg.eval__ = True or __cfg.workflow__ = [('val', 1)]
```
bash train.sh
```
Run the script then you can find the generated experimental logs in the folder `checkpoints`.

### testing
Follow the instructions below to begin testing our model.
**Step1.** Set model weights in configs/option_Net. The model weight file will be placed in `results/derain/Rain200H/Net/Test/***/weight.pth.tar`

>   model_path = f'' # model weight

**Step2.** Test the model.
```
python run_derain_test.py
```
Run the script then you can find the output visual results in the folder `results/derain/Rain200H/Net/Test/***/results/`.


## 🔧 Pre-trained Models and Results
| Datasets |                                                                     Pre-trained Models                                                                      |                                  Results                                  |
|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|
| Rain200L |                                          [Baidu Netdisk](https://pan.baidu.com/s/1RiwVA7z6pRiDcGn_MtS2hQ?pwd=1234)                                          | [Baidu Netdisk](https://pan.baidu.com/s/1DefDy0nWxfALudmv4i-oTg?pwd=1234) |
| Rain200H |                                         [Baidu Netdisk](https://pan.baidu.com/s/1vKEJYc9he3myGh_mizTXGg?pwd=1234)                                           | [Baidu Netdisk](https://pan.baidu.com/s/13a6lRGg9N8o1fCF0-7cHsQ?pwd=1234) |
| DID-Data |           [Baidu Netdisk](https://pan.baidu.com/s/1usmQ_GYFgYr0fWEh6NGrJQ?pwd=1234)|                                  Results                                  |
| DDN-Data |           [Baidu Netdisk](https://pan.baidu.com/s/1jf1g0nRXiRyjG7rBTM2qlQ?pwd=1234)  |                                  Results                                  |
| SPA-Data |           [Baidu Netdisk](https://pan.baidu.com/s/1VYIo3sNaONmEtNQv254L4w?pwd=1234)  |                                  Results                                  |


## 🚨 Performance Evaluation
See folder `matlab`

1) *for Rain200L/H and SPA-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](matlab/evaluate_PSNR_SSIM.m).

2) *for DID-Data and DDN-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](matlab/statistic.m).


## 👍 Acknowledgement
This code is based on the [DFTL](https://github.com/XiaoXiao-Woo/derain) and [MSDT](https://github.com/cschenhm/MSDT).


