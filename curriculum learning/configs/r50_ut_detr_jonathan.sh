#!/usr/bin/env bash
​
set -x
# Original labeled annotation file: 2151_ground_8605_aerial_train_aligned_ids_w_indicator.json
#EXP_DIR=results/pretrained8k2k/
#EXP_DIR=results/pretrained8k2k_unlabeledAerial/
#EXP_DIR=results/pretrained8k2k_unlabeled8k8k/  # 'unlabeled_g8k_a8k.json'
#EXP_DIR=results/pretrained8k2k_unlabeled8k2k/  # 'unlabeled_8k2k.json'
#EXP_DIR=results/pretrained8k2k_unlabeledall/  # 'unlabeled_all.json'  labeled:'8k_aerial_8k_ground_train_aligned_ids_w_points.json'
EXP_DIR=results/vanilla_omni_detr/  # labeled: '2151_ground_8605_aerial_train_aligned_ids_w_indicator.json', unlabeled: 'unlabeled_random_sample_21512.json'
                                        # labeled: 2151_aerial_2151_ground_train_aligned_ids_w_indicator.json
PY_ARGS=${@:1}
​
python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_train_label 'supervised_annotations/aerial/aligned_ids/aerial_train_aligned_ids_w_indicator_with_perspective_with_points.json' \
    --annotation_json_train_unlabel 'unsupervised_annotations/Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path '/data/SDU-GAMODv4/SDU-GAMODv4' \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --pixels 600 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 10 \
    --dataset_file 'dvd' \
    --resume '' \