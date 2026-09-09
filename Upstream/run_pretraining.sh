#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Edit only these variables for your environment.
DATA_DIR="/path/to/unlabeled_uwf_images"
INIT_CHECKPOINT="/path/to/initial_mae_or_retfound_checkpoint.pth"
OUTPUT_DIR="checkpoints/pretrain_uwf"
LOG_DIR="logs/pretrain_uwf"

python Upstream/core/train.py \
  --pretrain \
  --model mae_vit_large_patch16 \
  --input_size 448 \
  --batch_size 16 \
  --num_workers 8 \
  --blr 1e-3 \
  --epochs 800 \
  --warmup_epochs 50 \
  --data_path "$DATA_DIR" \
  --chkpt_path "$INIT_CHECKPOINT" \
  --task pretrain_uwf \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda
