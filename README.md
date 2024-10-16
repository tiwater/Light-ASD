## A Light Weight Model for Active Speaker Detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-light-weight-model-for-active-speaker/audio-visual-active-speaker-detection-on-ava)](https://paperswithcode.com/sota/audio-visual-active-speaker-detection-on-ava?p=a-light-weight-model-for-active-speaker)

This repository contains the code and model weights for our [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liao_A_Light_Weight_Model_for_Active_Speaker_Detection_CVPR_2023_paper.pdf) (CVPR 2023):

> A Light Weight Model for Active Speaker Detection  
> Junhua Liao, Haihan Duan, Kanghui Feng, Wanbing Zhao, Yanbing Yang, Liangyin Chen

***

## 执行

1. 在开发板上安装 Python 执行环境。注意：
   - RKNN-Toolkit Ubuntu 20.04 只支持 Python 3.8 和 3.9。
   - Ubuntu 22.04 支持 Python 3.10 和 3.11。

2. 将本项目拷贝至 RK3588 开发板，将 `models/rknn_models.zip` 解压至 `models` 目录。
 
3. 在开发板上执行以下命令以安装必要的库：
   ```bash
   sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc ffmpeg
   pip install -r requirements.txt
   ```

4. 获得瑞芯微官方的 RKNN-Toolkit：
   ```bash
   git clone https://hub.nuaa.cf/airockchip/rknn-toolkit2.git
   ```

9. 将 RKNN-Toolkit 安装到开发板上：
   ```bash
   cd rknn-toolkit2
   pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cpxx-cpxx-linux_aarch64.whl
   ```
   其中的 `cpxx` 需要替换为 Python 的版本号，请在 `rknn-toolkit-lite2/packages` 目录下查找和你的 Python 版本匹配的 `.whl` 文件。例如：
   ```bash
   pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cp38-cp38-linux_aarch64.whl
   ```

6. 执行：
   ```bash
   python inference_onnx.py
   ```
   利用 ONNX 模型推理

   或者
   ```bash
   python inference_rknn.py
   ```
   利用 RKNN 模型推理

## 模型转换

1. 如果需要重新生成模型，请执行以下步骤：
   
2. 参考 [RKNN-Toolkit2 快速入门指南](https://hub.nuaa.cf/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RV1106_RV1103_Quick_Start_RKNN_SDK_V2.0.0beta0_CN.pdf)，在 PC 端用 Docker 安装 RKNN-Toolkit2 镜像。

3. 启动镜像后，在镜像中执行以下步骤：

4. 利用 `docker cp` 将本项目源码拷贝进容器中。

5.  执行：
    ```bash
    python gen_asd_rknn.py
    ```
    会依次生成 LightASD 的 pt, onnx, rknn 模型。

6.  执行：
    ```bash
    python gen_loss_rknn.py
    ```
    会依次生成 LossAV 模型 的 pt, onnx, rknn 模型。

## 模型调试及评估

1. 执行：
    ```bash
    python evaluate_acc_asd.py
    ```
    评估 LightASD onnx 和 rknn 模型的精度差。

2. 执行：
    ```bash
    python evaluate_acc_loss.py
    ```
    评估 LossAV 模型的 onnx 和 rknn 模型的精度差。

3. 执行：
    ```bash
    python evaluate_perf.py
    ```
    评估 LightASD 模型的计算性能。

4. 执行：
    ```bash
    python evaluate_mem.py
    ```
    查看 LightASD 模型的内存占用情况。

5. 执行：
    ```bash
    python gen_quantize_data.py
    python quantize.py
    ```
    生成量化数据并执行量化。注意，量化数据是从 Columbia_test.py 解析出的文件中提取而来，注意阅读 gen_quantize_data.py 开始的注释。

***
### Evaluate on AVA-ActiveSpeaker dataset 

#### Data preparation
Use the following code to download and preprocess the AVA dataset.
```
python train.py --dataPathAVA AVADataPath --download 
```
The AVA dataset and the labels will be downloaded into `AVADataPath`.

#### Training
You can train the model on the AVA dataset by using:
```
python train.py --dataPathAVA AVADataPath
```

or
```
python train_2D.py --dataPathAVA AVADataPath
```
which is a customized version to support rknn.

`exps/exps1/score.txt`: output score file, `exps/exp1/model/model_00xx.model`: trained model, `exps/exps1/val_res.csv`: prediction for val set.

#### Testing
Our model weights have been placed in the `weight` folder. It performs `mAP: 94.06%` in the validation set. You can check it by using: 
```
python train.py --dataPathAVA AVADataPath --evaluation
```


***
### Evaluate on Columbia ASD dataset

#### Testing
The model weights trained on the AVA dataset have been placed in the `weight` folder. Then run the following code.
```
python Columbia_test.py --evalCol --colSavePath colDataPath
```
The Columbia ASD dataset and the labels will be downloaded into `colDataPath`. And you can get the following F1 result.
| Name |  Bell  |  Boll  |  Lieb  |  Long  |  Sick  |  Avg.  |
|----- | ------ | ------ | ------ | ------ | ------ | ------ |
|  F1  |  82.7% |  75.7% |  87.0% |  74.5% |  85.4% |  81.1% |

We have also provided the model weights fine-tuned on the TalkSet dataset. Due to space limitations, we did not exhibit it in the paper. Run the following code.
```
python Columbia_test.py --evalCol --pretrainModel weight/finetuning_TalkSet.model --colSavePath colDataPath
```
And you can get the following F1 result.
| Name |  Bell  |  Boll  |  Lieb  |  Long  |  Sick  |  Avg.  |
|----- | ------ | ------ | ------ | ------ | ------ | ------ |
|  F1  |  97.7% |  86.3% |  98.2% |  99.0% |  96.3% |  95.5% |


***
### An ASD Demo with pretrained Light-ASD model
You can put the raw video (`.mp4` and `.avi` are both fine) into the `demo` folder, such as `0001.mp4`. 
```
python Columbia_test.py --videoName 0001 --videoFolder demo
```
By default, the model loads weights trained on the AVA-ActiveSpeaker dataset. If you want to load weights fine-tuned on TalkSet, you can execute the following code.
```
python Columbia_test.py --videoName 0001 --videoFolder demo --pretrainModel weight/finetuning_TalkSet.model
```
You can obtain the output video `demo/0001/pyavi/video_out.avi`, where the active speaker is marked by a green box and the non-active speaker by a red box.


***
### Citation

Please cite our paper if you use this code or model weights. 

```
@InProceedings{Liao_2023_CVPR,
    author    = {Liao, Junhua and Duan, Haihan and Feng, Kanghui and Zhao, Wanbing and Yang, Yanbing and Chen, Liangyin},
    title     = {A Light Weight Model for Active Speaker Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22932-22941}
}
```

***
### Acknowledgments
Thanks for the support of TaoRuijie's open source [repository](https://github.com/TaoRuijie/TalkNet-ASD) for this research.