import os
from glob import glob
import re
from queue import Empty
import time
from typing import List, Tuple
import torch
import torch.multiprocessing as multiprocessing
import psutil
import numpy as np
import argparse
import cv2
import shutil
import hashlib
from tqdm import tqdm
from video_track import TrackingVideoImage, get_training_data


def process_video_with_flame(video_path, output_dir, device_id, target_fps=15, image_size=512, head_detect_freq=1, min_valid_frames=0):
    """Process a video using FLAME tracking and generate training data."""
    try:
        # Initialize tracker
        tracker = TrackingVideoImage(output_dir, device_id)

        # Process video directly - this will:
        # 1. Extract and preprocess frames (creates preprocess directory)
        # 2. Optimize FLAME parameters
        # 3. Export results
        # 4. Generate training data
        tracker.tracking_video(
            video_path=video_path,
            image_size=image_size,
            target_fps=target_fps,
            head_detect_freq=head_detect_freq,
            min_valid_frames=min_valid_frames
        )

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in process_video_with_flame: {e}")
        raise


def collect_video_tasks(input_base_path: str, output_base_path: str, action_seq: str = None, target_fps: int = 30):
    """Collect video-level tasks by processing one person at a time to reduce memory pressure."""
    tasks = []

    if not os.path.exists(input_base_path):
        print(f"Warning: Input base path does not exist: {input_base_path}")
        return tasks

    print("Collecting video files and extracting frames...")

    # Get all action sequences (directories starting with "sequence_")
    if action_seq is None:
        # If no specific action sequence is provided, find all sequences
        action_seq_dirs = [d for d in sorted(os.listdir(input_base_path))
                           if os.path.isdir(os.path.join(input_base_path, d)) and d.startswith('sequence_')]
    else:
        # Use the specified action sequence
        action_seq_dirs = [action_seq] if action_seq.startswith('sequence_') else [f'sequence_{action_seq}']

    print(f"Found {len(action_seq_dirs)} action sequences to process")

    for action_seq_dir in action_seq_dirs:
        sequence_path = os.path.join(input_base_path, action_seq_dir)
        if not os.path.exists(sequence_path):
            print(f"  Warning: Sequence path does not exist: {sequence_path}")
            continue

        print(f"Processing action sequence: {action_seq_dir}")

        # Get all person IDs in this sequence
        person_ids = [d for d in sorted(os.listdir(sequence_path))
                      if os.path.isdir(os.path.join(sequence_path, d))]
        print(f"  Found {len(person_ids)} persons in {action_seq_dir}")

        for person_id in person_ids:
            print(f"  Processing person {person_id}...")
            person_path = os.path.join(sequence_path, person_id)

            # Get all action sequence subdirectories for this person
            action_subdirs = [d for d in sorted(os.listdir(person_path))
                              if os.path.isdir(os.path.join(person_path, d))]
            print(f"    Found {len(action_subdirs)} action subdirs for person {person_id}")

            for action_subdir in action_subdirs:
                action_path = os.path.join(person_path, action_subdir)

                # Look for video files in this action directory
                video_files = glob(os.path.join(action_path, "*.mp4"))
                if not video_files:
                    print(f"      No video files found in {action_subdir} for person {person_id}")
                    continue

                print(f"      Found {len(video_files)} video files in {action_subdir}")

                # Process each video for this person and action
                for video_path in video_files:
                    cam_filename = os.path.basename(video_path)
                    cam_id = os.path.splitext(cam_filename)[0]  # e.g., "cam_222200037"

                    # Skip certain cameras if needed
                    skip_cams = {'cam_222200044', 'cam_222200042', 'cam_222200045', 'cam_221501007'}
                    if cam_id in skip_cams or "stacked" in cam_id:
                        print(f"        Skipping camera {cam_id} for person {person_id}")
                        continue

                    # Create output directory: output_base_path/person_id/cam_id/action_seq/
                    # Extract action sequence name (remove "sequence_" prefix if present)
                    action_seq_name = action_seq_dir.replace('sequence_', '') if action_seq_dir.startswith('sequence_') else action_seq_dir

                    output_dir = os.path.join(output_base_path, person_id, cam_id, action_seq_name)

                    # Check if already processed
                    if os.path.exists(output_dir):
                        # Check if FLAME parameters exist
                        flame_param_dir = os.path.join(output_dir, 'flame_param')
                        if os.path.exists(flame_param_dir):
                            flame_param_files = glob(os.path.join(flame_param_dir, '*.npz'))
                            if flame_param_files:
                                print(f"        Skipping {person_id}/{cam_id}/{action_seq_name} - already processed")
                                continue
                            else:
                                print(f"        FLAME parameters missing for {person_id}/{cam_id}/{action_seq_name}, will reprocess")
                        else:
                            print(f"        FLAME parameters missing for {person_id}/{cam_id}/{action_seq_name}, will reprocess")

                    # Add task: (action_seq_name, person_id, cam_id, video_path, output_dir)
                    tasks.append((action_seq_name, person_id, cam_id, video_path, output_dir))
                    print(f"        Added task for {person_id}/{cam_id}/{action_seq_name}")

            print(f"    Completed processing person {person_id}")

    return tasks


def collect_vfhq_tasks(input_base_path: str, output_base_path: str, action_seq: str = None, target_fps: int = 30, start_group: int = None, end_group: int = None):
    """Collect VFHQ tasks. 
    VFHQ structure: base/group/Clip+PersonID+.../00000000.png
    
    Output structure: output_base/PersonID/cam_000000000/EXP-0-random
    
    - PersonID: Parsed from clip folder name (e.g. Clip+MBFrb3EcJeA+... -> MBFrb3EcJeA).
    - CamID: Sequential per person (cam_000000000, cam_000000001, ...)
    - Action: fixed as 'EXP-0-random'.
    """
    tasks = []
    if not os.path.exists(input_base_path):
        print(f"Warning: Input base path does not exist: {input_base_path}")
        return tasks

    print("Collecting VFHQ tasks...")
    
    # Get all group directories using scantree for speed
    all_group_dirs = []
    with os.scandir(input_base_path) as it:
        for entry in it:
            if entry.is_dir():
                all_group_dirs.append(entry.name)
    all_group_dirs.sort()
    
    print(f"Found {len(all_group_dirs)} total groups")

    # Filter groups if range provided
    group_dirs = []
    for g_name in all_group_dirs:
        # Assuming format "group123"
        try:
            # Extract number from "group123" -> 123
            g_num = int(''.join(filter(str.isdigit, g_name)))
            
            if start_group is not None and g_num < start_group:
                continue
            if end_group is not None and g_num > end_group:
                continue
            group_dirs.append(g_name)
        except ValueError:
            print(f"Warning: Could not parse group number from {g_name}, skipping filtering for this folder")
            group_dirs.append(g_name) # Keep it if we can't parse number but it was in the list

    print(f"Processing {len(group_dirs)} groups after filtering (Start: {start_group}, End: {end_group})")

    # Dictionary to hold clips per person: { person_id: [clip_path1, clip_path2, ...] }
    person_clips_map = {}

    print("Scanning for persons and clips...")
    for group_dir in tqdm(group_dirs, desc="Scanning groups"):
        group_path = os.path.join(input_base_path, group_dir)
        
        # Use scandir for faster iteration
        clip_paths = []
        with os.scandir(group_path) as it:
            for entry in it:
                if entry.is_dir():
                    clip_paths.append((entry.name, entry.path))
        
        # Sort by Name
        clip_paths.sort(key=lambda x: x[0])
        
        for clip_name, clip_path in clip_paths:
            # Optional: Check emptiness quickly (optional, might slow down)
            # if not any(os.scandir(clip_path)): continue 
            
            # Parse PersonID from Clip Name
            # Format: Clip+PersonID+...
            parts = clip_name.split('+')
            if len(parts) > 1:
                person_id = parts[1]
                # Remove all special characters (keep only alphanumeric)
                person_id = re.sub(r'[^a-zA-Z0-9]', '', person_id)
            else:
                # print(f"Warning: Could not parse PersonID from {clip_name}, using 'unknown'")
                person_id = "unknown"
            
            if person_id not in person_clips_map:
                person_clips_map[person_id] = []
            # Store both name and path to generate hash later
            person_clips_map[person_id].append((clip_name, clip_path))

    print(f"Found {len(person_clips_map)} unique persons across selected groups")
    
    total_tasks = 0
    # Create tasks with Hash-based cam_id
    for person_id, clips in person_clips_map.items():
        # Sort clips just to be deterministic in iteration order (optional but good practice)
        clips.sort()
        
        for clip_name, clip_path in clips:
            # Generate deterministic hash from clip name
            # Use MD5 to get a unique identifier
            hash_object = hashlib.md5(clip_name.encode())
            # Take first 10 characters of hex digest for a concise unique ID
            cam_id_hash = hash_object.hexdigest()[:10]
            cam_id = f"cam_{cam_id_hash}"
            
            action_seq_name = "EXP-0-random"
            
            output_dir = os.path.join(output_base_path, person_id, cam_id, action_seq_name)
            
            # Check if already processed (checking flame_param existence)
            if os.path.exists(output_dir):
                flame_param_dir = os.path.join(output_dir, 'flame_param')
                if os.path.exists(flame_param_dir):
                        flame_param_files = glob(os.path.join(flame_param_dir, '*.npz'))
                        if flame_param_files:
                            # print(f"        Skipping {person_id}/{cam_id} (VFHQ) - already processed")
                            continue
            
            # Add task
            tasks.append((action_seq_name, person_id, cam_id, clip_path, output_dir))
            total_tasks += 1
            
    print(f"Collected {len(tasks)} VFHQ tasks")
    return tasks


def extract_frames_from_video(video_path: str, output_dir: str, target_fps: int = 30):
    """Extract frames from video and save them to the output directory."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Cannot open video: {video_path}")
            return False

        # Get video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        skip_ratio = orig_fps / target_fps
        frame_count = 0
        frame_idx = 0
        
        print(f"  Extracting frames from {os.path.basename(video_path)} (FPS: {orig_fps} -> {target_fps})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % round(skip_ratio) == 0:
                # Save frame
                frame_name = f'{frame_count:05d}.png'
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                frame_count += 1
            frame_idx += 1
        
        cap.release()
        print(f"  Extracted {frame_count} frames to {output_dir}")
        return True
        
    except Exception as e:
        print(f"  Failed to extract frames from {video_path}: {e}")
        return False


def distribute_tasks(tasks, num_gpus=8, num_workers=1, unavailable_gpus: List[int] = None):
    """Distribute video-level tasks across available GPUs."""
    if unavailable_gpus is None:
        unavailable_gpus = []
    
    available_gpus = sorted([i for i in range(num_gpus) if i not in unavailable_gpus])
    if not available_gpus:
        raise ValueError("No available GPUs to use!")
    
    print(f"Available GPUs: {available_gpus}")
    print(f"Number of video tasks: {len(tasks)}")
    
    tasks_per_gpu = len(tasks) // len(available_gpus)
    extra_tasks = len(tasks) % len(available_gpus)
    
    gpu_tasks = {gpu_id: [] for gpu_id in available_gpus}
    
    task_idx = 0
    for gpu_id in available_gpus:
        num_tasks = tasks_per_gpu + (1 if extra_tasks > 0 else 0)
        if extra_tasks > 0:
            extra_tasks -= 1
            
        gpu_tasks[gpu_id] = tasks[task_idx:task_idx + num_tasks]
        task_idx += num_tasks
    
    distributed = []
    for gpu_id in available_gpus:
        gpu_task_list = gpu_tasks[gpu_id]
        tasks_per_worker = len(gpu_task_list) // num_workers
        extra_worker_tasks = len(gpu_task_list) % num_workers
        
        start_idx = 0
        for worker_idx in range(num_workers):
            num_worker_tasks = tasks_per_worker + (1 if extra_worker_tasks > 0 else 0)
            if extra_worker_tasks > 0:
                extra_worker_tasks -= 1
                
            end_idx = start_idx + num_worker_tasks
            distributed.append(gpu_task_list[start_idx:end_idx])
            start_idx = end_idx

    for gpu_id in range(num_gpus):
        if gpu_id in gpu_tasks:
            task_count = len(gpu_tasks[gpu_id])
            print(f"GPU {gpu_id}: {task_count} video tasks")
        else:
            print(f"GPU {gpu_id}: UNAVAILABLE")
    
    return distributed, available_gpus


def set_process_priority():
    """Set the current process to a lower priority to prevent CPU overload."""
    try:
        p = psutil.Process(os.getpid())
        p.nice(10)  # Set to lower priority (higher nice value)
    except Exception as e:
        print(f"Failed to set process priority: {e}")


def clean_intermediate_data(output_dir):
    try:
        preprocess_path = os.path.join(output_dir, 'preprocess')
        if os.path.exists(preprocess_path):
            shutil.rmtree(preprocess_path)
        
        tracking_path = os.path.join(output_dir, 'tracking')
        if os.path.exists(tracking_path):
            shutil.rmtree(tracking_path)
        
        frames_path = os.path.join(output_dir, 'frames')
        if os.path.exists(frames_path):
            shutil.rmtree(frames_path)
        
        images_path = os.path.join(output_dir, 'images')
        if os.path.exists(images_path):
            shutil.rmtree(images_path)

        fg_mask_path = os.path.join(output_dir, 'fg_masks')
        if os.path.exists(fg_mask_path):
            shutil.rmtree(fg_mask_path)
                
    except Exception as e:
        print(f"Failed to clean intermediate data: {e}")


def worker(worker_id, gpu_id, task_queue, target_fps, head_detect_freq=1, min_valid_frames=0):
    """Worker function to process individual videos."""
    set_process_priority()

    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            # Check if the requested GPU exists
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"GPU {gpu_id} does not exist")

            # Set CUDA device
            torch.cuda.set_device(gpu_id)

            # Verify the device was set correctly
            if torch.cuda.current_device() != gpu_id:
                raise RuntimeError(f"Failed to set GPU {gpu_id}")

            # Test GPU with a small operation
            test_tensor = torch.zeros(1, device=f'cuda:{gpu_id}')
            del test_tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Clear cache after initialization

            print(f"[GPU {gpu_id}] Successfully initialized")
            break

        except Exception as e:
            retry_count += 1
            print(f"[GPU {gpu_id}] Initialization attempt {retry_count} failed: {e}")
            if retry_count < max_retries:
                time.sleep(1)  # Wait before retrying
            else:
                print(f"[GPU {gpu_id}] Failed to initialize after {max_retries} attempts")
                return

    task_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 3

    while True:
        try:
            # Check CPU usage before processing
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                print(f"[GPU {gpu_id}] CPU usage too high ({cpu_percent}%), waiting...")
                time.sleep(1)
                continue

            try:
                task = task_queue.get_nowait()
            except Empty:
                time.sleep(0.1)
                continue

            if task is None:  # End signal
                print(f"Worker {worker_id} on GPU {gpu_id} finished, processed {task_count} video tasks")
                break

            task_count += 1
            action_seq_name, person_id, cam_id, video_path, output_dir = task

            # Create output directory for this video
            os.makedirs(output_dir, exist_ok=True)

            # Check if already processed
            flame_param_dir = os.path.join(output_dir, 'flame_param')
            if os.path.exists(flame_param_dir):
                flame_param_files = glob(os.path.join(flame_param_dir, '*.npz'))
                if flame_param_files:
                    print(f"[GPU {gpu_id}] Skipping {person_id}/{cam_id}/{action_seq_name} - already processed")
                    continue
                else:
                    print(f"[GPU {gpu_id}] FLAME parameters missing for {person_id}/{cam_id}/{action_seq_name}, will reprocess")

            try:
                # Verify GPU is still available before processing
                if not torch.cuda.is_available() or torch.cuda.current_device() != gpu_id:
                    raise RuntimeError("GPU became unavailable")

                print(f"[GPU {gpu_id}] Processing {person_id}/{cam_id}/{action_seq_name}")

                # Process video directly with FLAME tracking
                # This will handle frame extraction, preprocessing, optimization, and export
                print(f"[GPU {gpu_id}] Processing video with FLAME...")
                process_video_with_flame(
                    video_path=video_path,
                    output_dir=output_dir,
                    device_id=gpu_id,
                    target_fps=target_fps,
                    head_detect_freq=head_detect_freq,
                    min_valid_frames=min_valid_frames
                )

                consecutive_errors = 0
                print(f"[GPU {gpu_id}] Successfully processed {person_id}/{cam_id}/{action_seq_name}")

                clean_intermediate_data(output_dir)  # Commented out to keep intermediate folders

                # Clear GPU memory after each successful processing
                torch.cuda.empty_cache()

            except Exception as e:
                consecutive_errors += 1
                print(f"[GPU {gpu_id}] Failed on {person_id}/{cam_id}/{action_seq_name}: {e}")

                # Clear GPU memory even on failure
                torch.cuda.empty_cache()

                if consecutive_errors >= max_consecutive_errors:
                    print(f"[GPU {gpu_id}] Too many consecutive errors ({consecutive_errors}), stopping worker")
                    break

        except Exception as e:
            print(f"[GPU {gpu_id}] Worker error: {e}")
            # Clear GPU memory on any error
            torch.cuda.empty_cache()
            continue

    # Final cleanup
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] Worker cleanup completed")


def launch(input_base_path: str, output_base_path: str, action_seq: str = None, num_gpus=8, num_workers=1,
           unavailable_gpus: List[int] = None, target_fps=30, head_detect_freq=1, dataset_type='nersemble', 
           start_group: int = None, end_group: int = None):
    """Launch distributed video-level processing with frame extraction and FLAME modeling."""
    set_process_priority()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    available_gpu_count = torch.cuda.device_count()
    if available_gpu_count < num_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpu_count} are available")
        num_gpus = available_gpu_count

    # Define minimal valid frames based on dataset type
    min_valid_frames = 32 if dataset_type == 'vfhq' else 0

    # Collect video-level tasks
    if dataset_type == 'nersemble':
        all_tasks = collect_video_tasks(input_base_path, output_base_path, action_seq, target_fps)
    elif dataset_type == 'vfhq':
        all_tasks = collect_vfhq_tasks(input_base_path, output_base_path, action_seq, target_fps, start_group, end_group)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    all_tasks = sorted(all_tasks, key=lambda x: (x[1], x[2], x[0]))  # Sort by person_id, cam_id, action_seq

    if not all_tasks:
        print("No video tasks found to process!")
        return

    print(f"Found {len(all_tasks)} video tasks to process")
    if min_valid_frames > 0:
        print(f"Enforcing minimum valid frames: {min_valid_frames}")

    ctx = multiprocessing.get_context('spawn')
    task_queue = ctx.Queue()

    distributed_tasks, available_gpus = distribute_tasks(all_tasks, num_gpus, num_workers, unavailable_gpus)

    # Put tasks in queue
    for tasks in distributed_tasks:
        for task in tasks:
            task_queue.put(task)

    # Put end signals
    for _ in range(len(available_gpus) * num_workers):
        task_queue.put(None)

    # Start processes
    processes = []
    try:
        for worker_id in range(len(available_gpus) * num_workers):
            gpu_id = available_gpus[worker_id % len(available_gpus)]
            p = ctx.Process(target=worker, args=(worker_id, gpu_id, task_queue, target_fps, head_detect_freq, min_valid_frames))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print("All video processing tasks completed!")

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        print("All processes stopped")
        raise
    except Exception as e:
        print(f"Error during processing: {e}")
        # Clean up processes on error
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        raise
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def test_collect_tasks():
    """Test function to verify task collection works with the new structure."""
    input_base_path = "YOUR_DATA_PATH"
    output_base_path = "/tmp/test_output"

    # Test collecting tasks
    tasks = collect_video_tasks(input_base_path, output_base_path, "EXP-5-mouth_part-4", target_fps=7.5)
    print(f"Found {len(tasks)} tasks")
    for task in tasks[:5]:  # Show first 5 tasks
        print(f"Task: {task}")
    return tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed video-level FLAME parameters pipeline with frame extraction')
    parser.add_argument('--input_base_path', type=str, required=True,
                        help='Input base path containing nersemble data (e.g., /path/to/nersemble_data/nersemble_data)')
    parser.add_argument('--output_base_path', type=str, required=True,
                        help='Output base path for processed results')
    parser.add_argument('--action_seq', type=str, default=None,
                        help='Specific action sequence to process (e.g., EXP-5-mouth_part-4). If not specified, process all sequences.')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers per GPU')
    parser.add_argument('--target_fps', type=float, default=15,
                        help='Target FPS for frame extraction')
    parser.add_argument('--head_detect_freq', type=int, default=1,
                        help='Frequency of head detection (detect every N frames)')
    parser.add_argument('--dataset_type', type=str, default='nersemble', choices=['nersemble', 'vfhq'],
                        help='Dataset type (nersemble or vfhq)')
    parser.add_argument('--unavailable_gpus', type=str, default="",
                        help="Comma-separated list of unavailable GPU IDs (e.g. '0,2,5')")
    parser.add_argument('--start_group', type=int, default=None,
                        help='Start group index for VFHQ dataset (inclusive)')
    parser.add_argument('--end_group', type=int, default=None,
                        help='End group index for VFHQ dataset (inclusive)')

    args = parser.parse_args()

    unavailable_gpus = []
    if args.unavailable_gpus.strip():
        unavailable_gpus = [int(x.strip()) for x in args.unavailable_gpus.split(',') if x.strip()]
    print(f"Unavailable GPUs: {unavailable_gpus}")
    print(f"Input base path: {args.input_base_path}")
    print(f"Output base path: {args.output_base_path}")
    print(f"Dataset type: {args.dataset_type}")
    if args.action_seq:
        print(f"Processing action sequence: {args.action_seq}")
    
    if args.dataset_type == 'vfhq':
        if args.start_group is not None:
             print(f"Processing groups starting from {args.start_group}")
        if args.end_group is not None:
             print(f"Processing groups up to {args.end_group}")

    launch(
        input_base_path=args.input_base_path,
        output_base_path=args.output_base_path,
        action_seq=args.action_seq,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        unavailable_gpus=unavailable_gpus,
        target_fps=args.target_fps,
        head_detect_freq=args.head_detect_freq,
        dataset_type=args.dataset_type,
        start_group=args.start_group,
        end_group=args.end_group
    ) 