#!/usr/bin/env bash

CFGS="./cfgs"
WEIGHTS="./weights"

CUDA_VISIBLE_DEVICES=$1 python detect_single.py \
    --model_config_path ${CFGS}/$2/yolov3-$2-test.cfg \
    --data_config_path ${CFGS}/$2/$2.data \
    --weights_path ${WEIGHTS}/$2/yolov3-$2_final.weights\
    --conf_thres 0.5 \
    --nms_thres 0.4 \
    --area_thres 0.0 \
    --use_cuda
