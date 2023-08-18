#!/usr/bin/env bash

set -x

EXP_DIR=results/gamod/6kG8kA/
PY_ARGS=${@:1}

export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file 'gamod' \
    --data_path '../../DATASET/' \
    --num_queries 900 \
    --nheads 16 \
    --epochs 39 \
    --num_feature_levels 4 \
    ${PY_ARGS}
    
    #--finetune True \
    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/gamod_r50_deformable_detr.sh
    # --resume 'results/gamod/6kG8kA/checkpoint0014.pth' \