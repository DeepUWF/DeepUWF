#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after encrypting both data/plain/train and
# data/plain/test with encrypt_dataset.py.
PROTECTED_DATA_DIR="/path/to/protected_dataset"
PRETRAINED_CHECKPOINT="/path/to/pretrained_mae_checkpoint.pth"
OUTPUT_DIR="checkpoints/ric_finetune"
LOG_DIR="logs/ric_finetune"

python Upstream/core/train.py \
  --finetune \
  --model vit_large_patch16 \
  --num_classes 2 \
  --input_size 448 \
  --batch_size 8 \
  --num_workers 8 \
  --blr 1e-3 \
  --epochs 50 \
  --warmup_epochs 10 \
  --drop_path 0.1 \
  --data_path "$PROTECTED_DATA_DIR" \
  --chkpt_path "$PRETRAINED_CHECKPOINT" \
  --task ric_finetune \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda
