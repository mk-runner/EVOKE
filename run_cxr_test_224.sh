#!/bin/bash

python main_224.py \
--task "test" \
--data_name mimic_cxr \
--mimic_cxr_ann_path "MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json" \
--ft_monitor_metric RCB \
--version ft_v0411_v0416 \
--max_seq_len 100 \
--epochs 50 \
--load "https://huggingface.co/datasets/MK-runner/Multi-view-CXR/blob/main/checkpoints/MIMIC-CXR/resolution-224x224/model_best.pth" \
--is_multiview_learning "yes" \
--is_add_indication "yes" \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--num_layers 3 \
--optim RAdam \
--batch_size 32
