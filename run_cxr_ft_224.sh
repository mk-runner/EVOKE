#!/bin/bash

python main_224.py \
--task finetune \
--data_name mimic_cxr \
--mimic_cxr_ann_path "/home/20031211471/Code/Code/Data/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json" \
--ft_monitor_metric RCB \
--version ft_v0411_v0416 \
--max_seq_len 100 \
--epochs 50 \
--load "/home/20031211471/Code/Code/Second-Multiview/results/mimic_cxr/pretrain/v0411_v0416_pt/checkpoint/model_best.pth" \
--is_multiview_learning \
--is_add_indication \
--sk_type keywords \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--sk_topk 0 \
--num_layers 3 \
--optim RAdam \
--is_save_checkpoint \
--batch_size 32