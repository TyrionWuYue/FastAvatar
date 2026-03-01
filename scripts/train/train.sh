#!/bin/bash

set -e  # Exit on any error

ACC_CONFIG="./configs/train/accelerate-train-8gpu.yaml"
TRAIN_CONFIG="./configs/train/train.yaml"

# Step 1: Generate frame groups
echo "Step 1: Generating frame groups..."
python scripts/generate_frame_groups.py --config $TRAIN_CONFIG
if [ $? -ne 0 ]; then
    echo "Frame groups generation failed!"
    exit 1
fi
echo "Frame groups generation completed!"
echo

# Step 2: Start training
echo "Step 2: Starting training ..."
accelerate launch --config_file $ACC_CONFIG --main_process_port 0 -m FastAvatar.launch train.fastavatar \
    --config $TRAIN_CONFIG \