cd YOUR_PATH
PYTHONPATH=YOUR_PATH python data_preprocessing/distributed_video_tracking.py \
    --input_base_path SOURCE_DATA_PATH \
    --output_base_path TARGET_DATA_PATH \
    --action_seq YOUR_ACTION_SEQ \
    --target_fps 15 \
    --num_gpus 8 \
    --num_workers 2 \
    --head_detect_freq 4 \
    --dataset_type nersemble