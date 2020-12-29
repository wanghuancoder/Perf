# NGC PyTorch Mask R-CNN 性能测试

此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN) 实现的 Mask R-CNN 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC Pytorch Mask R-CNN 性能测试](#ngc-pytorch-mask-r-cnn-性能测试)
  - [一、环境介绍](#一环境介绍)
    - [1.物理机环境](#1物理机环境)
    - [2.Docker 镜像](#2docker-镜像)
  - [二、环境搭建](#二环境搭建)
    - [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
    - [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
  - [三、测试步骤](#三测试步骤)
    - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
    - [2.多机（32卡）测试](#2多机32卡测试)
  - [四、测试结果](#四测试结果)
  - [五、日志数据](#五日志数据)

## 一、环境介绍

### 1.物理机环境

我们使用了与Paddle测试完全相同的物理机环境：

- 单机（单卡、8卡）
  - 系统：CentOS Linux release 7.5.1804
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
  - Driver Version: 450.80.02
  - 内存：432 GB

- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

我们使用 NGC Pytorch 的代码仓库提供的脚本制作镜像：

- Docker: nvcr.io/nvidia/pytorch:20.06-py3
- PyTorch：1.6.0a0+9907a3e
- 模型代码：[NVIDIA/DeepLearningExamples/Pytorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)
- CUDA：11
- cuDNN：8.0.1

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- 下载NGC PyTorch repo,并进入目录

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples/PyTorch/Segmentation/MaskRCNN
   # 本次测试是在如下版本下完成的：
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
   ```

- 制作Docker镜像

   ```bash
   cd pytorch/
   bash scripts/docker/build.sh
   ```

- 启动Docker

   ```bash
   # 假设下载好的coco数据放在<path to data>目录下
   bash scripts/docker/interactive.sh <path to data>
   ```
   
### 2.多机（32卡）环境搭建

TODO Distribute

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现。其公开的测试报告请见：[《Mask R-CNN For PyTorch》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)。我们参照NGC公开的train_benchmark.sh编写了测试脚本，其中有三处不同：
1. 修改了`configs/e2e_mask_rcnn_R_50_FPN_1x.yaml`配置，将`DATASETS`的配置修改为：`TRAIN: ("coco_2014_train", "coco_2014_val")  TEST: ("coco_2014",)`。因为我们的数据集中并没有minusminival数据。
2. 测试中我们发现，如果`SOLVER.BASE_LR`为0.04，很多场景会出现loss为NAN的情况。NGC提供的0.04应该是为8卡每卡BatchSize为4准备的。因此，我们视情况调整了`SOLVER.BASE_LR`。
3. 测试中我们发现，有些场景运行中会core，报`OSError: image file is truncated`,我们在`/opt/conda/lib/python3.6/site-packages/torchvision/datasets/coco.py`中插入了两行代码解决该问题：
  ```
  from PIL import ImageFile
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  ```

- 下载我们编写的测试脚本，并执行该脚本

   ```bash
   wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/MaskRCNN/OtherReports/PyTorch/scripts/pytorch_test_all.sh
   wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/MaskRCNN/OtherReports/PyTorch/scripts/pytorch_test_base.sh
   bash pytorch_test_all.sh
   ```

- 执行后将得到如下日志文件：

   ```bash
   ./pytorch_gpu1_fp32_bs2.txt
   ./pytorch_gpu1_fp32_bs4.txt
   ./pytorch_gpu1_amp_bs2.txt
   ./pytorch_gpu1_amp_bs4.txt
   ./pytorch_gpu8_fp32_bs2.txt
   ./pytorch_gpu8_fp32_bs4.txt
   ./pytorch_gpu8_amp_bs2.txt
   ./pytorch_gpu8_amp_bs4.txt
   ```

### 2.多机（32卡）测试

TODO Distribute

## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=2) | FP32(BS=4) | AMP(BS=2) | AMP(BS=4)|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | 9.2 | 9.8 | 10.4 | 12.5 |
|8 | - | - | - | - |
|32 | - | - | - | - |

## 五、日志数据
- [1卡 FP32 BS=2 日志](./logs/pytorch_gpu1_fp32_bs2.txt)
- [1卡 FP32 BS=4 日志](./logs/pytorch_gpu1_fp32_bs4.txt)
- [1卡 AMP BS=2 日志](./logs/pytorch_gpu1_amp_bs2.txt)
- [1卡 AMP BS=4 日志](./logs/pytorch_gpu1_amp_bs4.txt)
- [8卡 FP32 BS=2 日志](./logs/pytorch_gpu8_fp32_bs2.txt)
- [8卡 FP32 BS=4 日志](./logs/pytorch_gpu8_fp32_bs4.txt)
- [8卡 AMP BS=2 日志](./logs/pytorch_gpu8_amp_bs2.txt)
- [8卡 AMP BS=4 日志](./logs/pytorch_gpu8_amp_bs4.txt)
