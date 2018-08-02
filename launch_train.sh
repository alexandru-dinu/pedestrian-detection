#!/usr/bin/env bash

CFGS="./cfgs"
WEIGHTS="./weights"
TRAINABLE_WEIGHTS="./weights/darknet53.conv.74"

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --num_epochs 10 \
    --model_config_path ${CFGS}/$2/yolov3-$2.cfg \
    --data_config_path ${CFGS}/$2/$2.data \
    --weights_path ${TRAINABLE_WEIGHTS} \
    --freeze_point 75 \
    --n_cpu 1 \
    --avg_interval 1 \
    --checkpoint_interval 1 \
    --checkpoint_dir checkpoints \
    --use_cuda \
    --shuffle
