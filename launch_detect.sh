#!/usr/bin/env bash

CFGS="./cfgs"
WEIGHTS="./weights"


CUDA_VISIBLE_DEVICES=$1 python detect.py \
    --model_config_path ${CFGS}/$2/yolov3-$2-test.cfg \
    --data_config_path ${CFGS}/$2/$2.data \
    --weights_path ${WEIGHTS}/$2/yolov3-$2_final.weights \
    --n_cpu 1 \
    --detect_dir citypersons \
    --conf_thres 0.6 \
    --nms_thres 0.4 \
    --use_cuda \
    --batch_count 10 \
    --shuffle
