#!/bin/bash

python main_384.py \
--task 'test' \
--data_name mimic_cxr \
--mimic_cxr_ann_path "MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json" \
--ft_monitor_metric RCB \
--version ft_v0623_large \
--max_seq_len 100 \
--epochs 50 \
--load "EVOKE/results/mimic_cxr/finetune/ft_v0623_large/checkpoint/model_best.pth" \
--is_multiview_learning "yes" \
--is_add_indication "yes" \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--num_layers 3 \
--optim RAdam \
--is_save_checkpoint "no" \
--batch_size 32