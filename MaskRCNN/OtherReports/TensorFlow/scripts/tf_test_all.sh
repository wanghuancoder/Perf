#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 8 --batch_size 2 > tf_gpu8_fp32_bs2.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 8 --batch_size 4 > tf_gpu8_fp32_bs4.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 8 --batch_size 2 --amp > tf_gpu8_amp_bs2.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 8 --batch_size 4 --amp > tf_gpu8_amp_bs4.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 1 --batch_size 2 > tf_gpu1_fp32_bs2.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 1 --batch_size 4 > tf_gpu1_fp32_bs4.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 1 --batch_size 2 --amp > tf_gpu1_amp_bs2.txt
rm -rf /tmp/model/*
python scripts/benchmark_training.py --gpus 1 --batch_size 4 --amp > tf_gpu1_amp_bs4.txt

