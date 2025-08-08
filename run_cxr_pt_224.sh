#!/bin/bash

python main_224.py \
--task pretrain \
--data_name mimic_cxr \
--mimic_cxr_ann_path "/Data/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json" \
--version v0411_v0416_pt \
--max_seq_len 100 \
--epochs 50 \
--lr 5.0e-5 \
--optim AdamW \
--is_save_checkpoint "yes" \
--is_multiview_learning "yes" \
--batch_size 32
