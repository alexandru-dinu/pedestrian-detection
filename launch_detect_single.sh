#!/usr/bin/env bash

CFGS="/home/tempuser/workspace/dinu-rotaru/darknet/own_cfg"
WEIGHTS="/home/tempuser/workspace/dinu-rotaru/darknet-weights"
DEFAULT_WEIGHTS="/home/tempuser/workspace/dinu-rotaru/darknet/weights/yolov3.weights"


CUDA_VISIBLE_DEVICES=$1 python detect_single.py \
    --model_config_path ${CFGS}/$2/yolov3-$2-test.cfg \
    --data_config_path ${CFGS}/$2/$2.data \
    --weights_path ${WEIGHTS}/$2/yolov3-$2_final.weights \
    --conf_thres 0.3 \
    --nms_thres 0.3 \
    --use_cuda
