#!/bin/bash

python main_384.py \
--task pretrain \
--data_name mimic_cxr \
--mimic_cxr_ann_path "Data/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json" \
--version v0623_pt_large \
--max_seq_len 100 \
--epochs 50 \
--pt_lr 5.0e-5 \
--ft_lr 5.0e-5 \
--sk_type keywords \
--multiple_positives_contrastive supcon \
--optim AdamW \
--is_save_checkpoint \
--is_multiview_learning \
--batch_size 32