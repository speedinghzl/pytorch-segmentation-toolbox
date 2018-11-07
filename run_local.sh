#!/bin/bash
uname -a
#date
#env
date

CS_PATH=$1
LR=1e-2
WD=5e-4
BS=8
STEPS=40000
GPU_IDS=0,1,2,3

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
python train.py --data-dir ${CS_PATH} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS}
python evaluate.py --data-dir ${CS_PATH} --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0