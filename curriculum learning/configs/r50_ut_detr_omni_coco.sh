#!/usr/bin/env bash

set -x

EXP_DIR=path to dir
PY_ARGS=${@:1}
echo ${PY_ARGS}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --data_path ' ' \
    --label_label_root ''\
    --unlabel_label_root ''\
    --annotation_json_train_label '' \
    --annotation_json_train_unlabel '' \
    --annotation_json_val '' \
    --data_dir_label_train '' \
    --data_dir_label_val '' \
    --data_dir_unlabel_train '' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --lr 2e-4 \
    --epoch 1000 \
    --lr_drop 150 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'dvd' \
    --resume 'results/vanilla_omni_detr_rajat/checkpoint0019.pth' 
 #done with json files having indicator name
##when doing val use alogn trained with indicator
# --BURN_IN_STEP 20 \
    # --TEACHER_UPDATE_ITER 1 \
    # --EMA_KEEP_RATE 0.9996 \
    # --data_path '/data/SDU-GAMODv4/SDU-GAMODv4' \
    # --label_label_root 'supervised_annotations/'\
    # --unlabel_label_root '/data/SDU-GAMODv4/SDU-GAMODv4/unsupervised_annotations/'\
    # --annotation_json_train_label '2151_ground_8605_aerial_train_aligned_ids_w_indicator.json' \
    # --annotation_json_train_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
    # --annotation_json_val 'aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json' \
    # --data_dir_label_train 'scaled_dataset/train/combinedview' \
    # --data_dir_label_val 'scaled_dataset/val/droneview' \
    # --data_dir_unlabel_train 'aerial/' \
    # --CONFIDENCE_THRESHOLD 0.7 \
    # --lr 2e-4 \
    # --epochs 1000 \
    # --lr_drop 150 \
    # --num_queries 900 \
    # --nheads 16 \
    # --num_feature_levels 4 \
    # --save_freq 10\
    # --dataset_file 'dvd' \