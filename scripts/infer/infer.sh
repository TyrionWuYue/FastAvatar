#!/bin/bash

# Core parameters
TRAIN_CONFIG="configs/inference/infer.yaml"
MODEL_NAME="model_zoo/fastavatar/"
IMAGE_INPUT="/inspire/hdd/project/project-public/fengkairui-25026/nersemble_data/sequence_EXP-1-head_part-7/325/EXP-1-head/cam_222200037.mp4"
MOTION_SEQS_DIR="assets/sample_motion/export/Donald_Trump/"
INFERENCE_N_FRAMES=4
MAX_SINGLE_FRAME_RENDER=8
MODE="Monocular"  # Options: "Monocular", "MultiView"

# Allow command line overrides
TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}
INFERENCE_N_FRAMES=${5:-$INFERENCE_N_FRAMES}
MAX_SINGLE_FRAME_RENDER=${6:-$MAX_SINGLE_FRAME_RENDER}
MODE=${7:-$MODE}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"
echo "INFERENCE_N_FRAMES: $INFERENCE_N_FRAMES"
echo "MAX_SINGLE_FRAME_RENDER: $MAX_SINGLE_FRAME_RENDER"
echo "MODE: $MODE"

# Essential parameters
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=7.5
EXPORT_VIDEO=true
EXPORT_MESH=true

# Hardware settings
device=0

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run inference
CUDA_VISIBLE_DEVICES=$device python -m FastAvatar.launch infer.fastavatar \
    --config $TRAIN_CONFIG \
    model_name=$MODEL_NAME \
    image_input=$IMAGE_INPUT \
    export_video=$EXPORT_VIDEO \
    export_mesh=$EXPORT_MESH \
    motion_seqs_dir=$MOTION_SEQS_DIR \
    render_fps=$RENDER_FPS \
    motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
    inference_N_frames=$INFERENCE_N_FRAMES \
    max_single_frame_render=$MAX_SINGLE_FRAME_RENDER \
    mode=$MODE