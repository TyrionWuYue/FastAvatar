#!/bin/bash

# step1. set TRAIN_CONFIG path to config file
TRAIN_CONFIG="configs/inference/infer.yaml"
MODEL_NAME="model_zoo/mine/"
IMAGE_INPUT="assets/sample_input/imgs/8frame"
MOTION_SEQS_DIR="assets/sample_motion/export/Donald_Trump/"

# Allow command line overrides
TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"

# Default parameters
MOTION_IMG_DIR=null
SAVE_PLY=false
SAVE_IMG=false
VIS_MOTION=false
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=true
EXPORT_MESH=true
CROSS_ID=false
TEST_SAMPLE=false
GAGA_TRACK_TYPE=""

# Hardware settings
device=2
nodes=0
if_multi_frames_compare=true

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run inference
CUDA_VISIBLE_DEVICES=$device python -m VGGTAvatar.launch infer.vggt_avatar \
    --config $TRAIN_CONFIG \
    model_name=$MODEL_NAME \
    image_input=$IMAGE_INPUT \
    export_video=$EXPORT_VIDEO \
    export_mesh=$EXPORT_MESH \
    motion_seqs_dir=$MOTION_SEQS_DIR \
    motion_img_dir=$MOTION_IMG_DIR \
    vis_motion=$VIS_MOTION \
    motion_img_need_mask=$MOTION_IMG_NEED_MASK \
    render_fps=$RENDER_FPS \
    motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
    save_ply=$SAVE_PLY \
    save_img=$SAVE_IMG \
    gaga_track_type=$GAGA_TRACK_TYPE \
    cross_id=$CROSS_ID \
    test_sample=$TEST_SAMPLE \
    rank=$device \
    nodes=$nodes \
    if_multi_frames_compare=$if_multi_frames_compare