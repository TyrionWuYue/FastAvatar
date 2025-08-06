#!/bin/bash

# step1. set TRAIN_CONFIG path to config file
TRAIN_CONFIG="configs/inference/infer.yaml"
MODEL_NAME="model_zoo/fastavatar/"
IMAGE_INPUT="/inspire/hdd/project/project-public/fengkairui-25026/nersemble_data/sequence_EXP-1-head_part-7/325/EXP-1-head/cam_222200037.mp4"
MOTION_SEQS_DIR="assets/sample_motion/export/Donald_Trump/"
INFERENCE_N_FRAMES=4
MAX_SINGLE_FRAME_RENDER=8

# Allow command line overrides
TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}
INFERENCE_N_FRAMES=${5:-$INFERENCE_N_FRAMES}
MAX_SINGLE_FRAME_RENDER=${6:-$MAX_SINGLE_FRAME_RENDER} 

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"
echo "INFERENCE_N_FRAMES: $INFERENCE_N_FRAMES"
echo "MAX_SINGLE_FRAME_RENDER: $MAX_SINGLE_FRAME_RENDER"

# Default parameters
MOTION_IMG_DIR=null
SAVE_PLY=false
SAVE_IMG=false
VIS_MOTION=false
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=7.5
EXPORT_VIDEO=true
EXPORT_MESH=true
CROSS_ID=false
TEST_SAMPLE=false
GAGA_TRACK_TYPE=""

# Hardware settings
device=0
nodes=0
if_multi_frames_compare=false

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
    motion_img_dir=$MOTION_IMG_DIR \
    vis_motion=$VIS_MOTION \
    motion_img_need_mask=$MOTION_IMG_NEED_MASK \
    render_fps=$RENDER_FPS \
    motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
    inference_N_frames=$INFERENCE_N_FRAMES \
    max_single_frame_render=$MAX_SINGLE_FRAME_RENDER \
    save_ply=$SAVE_PLY \
    save_img=$SAVE_IMG \
    gaga_track_type=$GAGA_TRACK_TYPE \
    cross_id=$CROSS_ID \
    test_sample=$TEST_SAMPLE \
    rank=$device \
    nodes=$nodes \
    if_multi_frames_compare=$if_multi_frames_compare