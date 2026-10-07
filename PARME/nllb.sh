#!/bin/bash

set -euo pipefail

export PYTHONNOUSERSITE=1


python -u run_translation.py \
  --model_name_or_path ./nllb_extended \
  --do_train \
  --train_file ./data/poj-eng/CPT080_train.jsonl \
  --output_dir ./nllb_finetuned_poj_080_3 \
  --per_device_train_batch_size 64 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --warmup_ratio 0.15 \
  --bf16 \
  --group_by_length \
  --logging_dir ./runs \
  --logging_strategy steps \
  --logging_steps 100 \
  --logging_first_step \
  --save_strategy steps \
  --save_steps 500 \
  --max_source_length 192 \
  --max_target_length 192 \
  --num_beams 5 \
  --weight_decay 0.01 \
  --seed 42 \
  --report_to tensorboard

echo "Job finished successfully!"