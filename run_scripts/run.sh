#!/usr/bin/env bash
CONFIG=/home/liu/ytx/road_damage/configs/ACTL_final.py
WORK_DIR=/home/liu/ytx/road_damage/outputs



python -m torch.distributed.launch --nproc_per_node=1 --master_port=9500 \
 tools/train.py ${CONFIG} --seed 0 --work-dir=${WORK_DIR} --launcher pytorch