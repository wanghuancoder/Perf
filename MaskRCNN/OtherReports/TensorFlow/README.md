# NGC TensorFlow2 Mask R-CNN 性能测试

此处给出了基于 [NGC TensorFlow2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) 实现的 Mask R-CNN 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC TensorFlow2 Mask R-CNN 性能测试](#ngc-tensorflow2-mask-r-cnn-性能测试)
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

我们使用 NGC TensorFlow2 的代码仓库提供的脚本制作镜像：

- Docker: nvcr.io/nvidia/tensorflow:20.06-tf2-py3
- TensorFlow：2.2.0
- 模型代码：[NVIDIA/DeepLearningExamples/TensorFLow2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN)
- CUDA：11
- cuDNN：8.0.1

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 NGC TensorFlow2 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- 下载NGC TensorFlow2 repo,并进入目录

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
   # 本次测试是在如下版本下完成的：
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
   ```

- 制作Docker镜像

   ```bash
   bash ./scripts/docker/build_tf2.sh
   ```

- 启动Docker

   ```bash
   # 即将制作的数据或制作完成的数据放在<path to tfrecords data>目录下
   bash ./scripts/docker/launch_tf2.sh <path to tfrecords data>
   ```
   
- 下载制作数据

  ```bash
  cd dataset
  bash download_and_preprocess_coco.sh /data
  #如果有已经下载好的coco数据集，可以放到/data/raw-data/下，注释掉download_and_preprocess_coco.sh脚本中download相关行，然后再执行脚本。可以省去下载时间。
  ```

### 2.多机（32卡）环境搭建

TODO Distribute

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《Mask R-CNN For Tensorflow》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN)

- 下载我们编写的测试脚本，并执行该脚本

   ```bash
   wget https://raw.githubusercontent.com/PaddlePaddle/Perf/master/MaskRCNN/OtherReports/TensorFlow/scripts/tf_test_all.sh
   bash tf_test_all.sh
   ```

- 执行后将得到如下日志文件：

   ```bash
   ./tf_gpu1_fp32_bs2.txt
   ./tf_gpu1_fp32_bs4.txt
   ./tf_gpu1_amp_bs2.txt
   ./tf_gpu1_amp_bs4.txt
   ./tf_gpu8_fp32_bs2.txt
   ./tf_gpu8_fp32_bs4.txt
   ./tf_gpu8_amp_bs2.txt
   ./tf_gpu8_amp_bs4.txt
   ```

### 2.多机（32卡）测试

TODO Distribute

## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=2) | FP32(BS=4) | AMP(BS=2) | AMP(BS=4)|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | 5.3 | 5.8 | 8.1 | 9.9 |
|8 | 37.8 | 39.3 | 49.6 | 54.8 |
|32 | - | - | - | - |

## 五、日志数据
- [1卡 FP32 BS=2 日志](./logs/tf_gpu1_fp32_bs2.txt)
- [1卡 FP32 BS=4 日志](./logs/tf_gpu1_fp32_bs4.txt)
- [1卡 AMP BS=2 日志](./logs/tf_gpu1_amp_bs2.txt)
- [1卡 AMP BS=4 日志](./logs/tf_gpu1_amp_bs4.txt)
- [8卡 FP32 BS=2 日志](./logs/tf_gpu8_fp32_bs2.txt)
- [8卡 FP32 BS=4 日志](./logs/tf_gpu8_fp32_bs4.txt)
- [8卡 AMP BS=2 日志](./logs/tf_gpu8_amp_bs2.txt)
- [8卡 AMP BS=4 日志](./logs/tf_gpu8_amp_bs4.txt)
