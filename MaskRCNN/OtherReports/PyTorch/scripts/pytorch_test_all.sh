#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


bash pytorch_test_base.sh 8 16 float32 > pytorch_gpu8_fp32_bs2.txt
bash pytorch_test_base.sh 8 32 float32 > pytorch_gpu8_fp32_bs4.txt
bash pytorch_test_base.sh 8 16 float16 > pytorch_gpu8_amp_bs2.txt
bash pytorch_test_base.sh 8 32 float16 > pytorch_gpu8_amp_bs4.txt
bash pytorch_test_base.sh 1 2 float32 > pytorch_gpu1_fp32_bs2.txt
bash pytorch_test_base.sh 1 4 float32 > pytorch_gpu1_fp32_bs4.txt
bash pytorch_test_base.sh 1 2 float16 > pytorch_gpu1_amp_bs2.txt
bash pytorch_test_base.sh 1 4 float16 > pytorch_gpu1_amp_bs4.txt

