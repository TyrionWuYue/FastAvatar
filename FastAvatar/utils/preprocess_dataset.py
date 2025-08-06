#!/usr/bin/env python3
"""
Dataset preprocessing utility for VGGTAvatar.
This script generates dataset JSON files from raw data directories.
"""

import os
import json
import numpy as np
import argparse
from typing import Dict, List, Union
from pathlib import Path


def process_root_directory(
    root_dir: str,
    frames_per_sample: int = 32,
    seed: int = 42,
    task_type: str = "monocular",
    max_input_frames: int = 16,
    samples_multiplier: int = 5
) -> Dict[str, Dict[str, List[Dict[str, Union[str, int]]]]]:
    """
    Process the root directory and generate frame groups based on task type.
    
    Directory structure:
    root_dir/
        sequence_task_part-x/
            person_id/
                cam_id/
                    processed_data/
                        00000/
                            rgb.npy
                            mask.npy
                            intrs.npy
                            landmark2d.npz
                            bg_color.npy
                        00001/
                            ...
    
    Args:
        root_dir: Root directory containing the dataset
        frames_per_sample: Number of camera-frame pairs per sample
        seed: Random seed for frame selection
        task_type: Type of task - "monocular", "multi-view", or "unified"
        max_input_frames: Maximum number of input frames (default: 16)
    
    Returns:
        Dictionary containing frame groups
        Format: {
            "sequence_name/person_id/group_id": {
                "data": [
                    {"camera": "cam1", "frame": 1},
                    {"camera": "cam1", "frame": 2},
                    ...
                ],
                "input_frames": 5
            }
        }
    """
    np.random.seed(seed)
    
    if task_type not in ["monocular", "multi-view", "unified"]:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'monocular', 'multi-view', or 'unified'")
    
    frame_groups: Dict[str, Dict[str, List[Dict[str, Union[str, int]]]]] = {}
    required_files = {'rgb.npy', 'mask.npy', 'intrs.npy', 'landmark2d.npz', 'bg_color.npy'}
    
    print(f"Processing root directory: {root_dir}")
    print(f"Task type: {task_type}")
    print(f"Frames per sample: {frames_per_sample}")
    
    total_sequences = 0
    total_persons = 0
    total_monocular_groups = 0
    total_multiview_groups = 0
    
    for sequence_name in sorted(os.listdir(root_dir)):
        # Filter out sequences containing "head" in their name
        if "head" in sequence_name.lower():
            print(f"Skipping sequence with 'head' in name: {sequence_name}")
            continue
            
        sequence_dir = os.path.join(root_dir, sequence_name)
        if not os.path.isdir(sequence_dir):
            continue
            
        total_sequences += 1
        
        for person_id in sorted(os.listdir(sequence_dir)):
            person_dir = os.path.join(sequence_dir, person_id)
            if not os.path.isdir(person_dir):
                continue
                
            total_persons += 1
            
            # Collect frames from all cameras for this person
            person_camera_frames = {}
            
            for camera_id in sorted(os.listdir(person_dir)):
                camera_dir = os.path.join(person_dir, camera_id)
                if not os.path.isdir(camera_dir):
                    continue
                    
                processed_data_root = os.path.join(camera_dir, "processed_data")
                if not os.path.isdir(processed_data_root):
                    continue
                    
                # Collect valid frames for this camera
                camera_frames = []
                frame_dirs = [d for d in sorted(os.listdir(processed_data_root)) 
                            if d.isdigit() and os.path.isdir(os.path.join(processed_data_root, d))]
                
                for d in frame_dirs:
                    frame_path = os.path.join(processed_data_root, d)
                    frame_files = set(os.listdir(frame_path))
                    if required_files.issubset(frame_files):
                        camera_frames.append(int(d))
                
                if len(camera_frames) > 0:
                    person_camera_frames[camera_id] = camera_frames
            
            if len(person_camera_frames) == 0:
                continue
            
            # Generate groups based on task type
            if task_type == "monocular":
                groups = _generate_monocular_groups(person_camera_frames, frames_per_sample, max_input_frames, samples_multiplier)
                mono_groups = groups
                multi_groups = []
            elif task_type == "multi-view":
                groups = _generate_multiview_groups(person_camera_frames, frames_per_sample, max_input_frames, samples_multiplier)
                mono_groups = []
                multi_groups = groups
            elif task_type == "unified":
                mono_groups = _generate_monocular_groups(person_camera_frames, frames_per_sample, max_input_frames, samples_multiplier)
                multi_groups = _generate_multiview_groups(person_camera_frames, frames_per_sample, max_input_frames, samples_multiplier)
                groups = mono_groups + multi_groups
            
            # Add groups to frame_groups and count them
            base_key = f"{sequence_name}/{person_id}"
            for group_idx, group_data in enumerate(groups, 1):
                key = f"{base_key}/{group_idx:05d}"
                frame_groups[key] = group_data  # group_data now contains both "data" and "input_frames"
            
            # Update counters
            total_monocular_groups += len(mono_groups)
            total_multiview_groups += len(multi_groups)
    
    # Print results based on task type
    if task_type == "monocular":
        print(f"Processed {total_sequences} sequences, {total_persons} persons, generated {total_monocular_groups} monocular frame groups")
    elif task_type == "multi-view":
        print(f"Processed {total_sequences} sequences, {total_persons} persons, generated {total_multiview_groups} multi-view frame groups")
    elif task_type == "unified":
        total_groups = total_monocular_groups + total_multiview_groups
        print(f"Processed {total_sequences} sequences, {total_persons} persons:")
        print(f"  - Generated {total_monocular_groups} monocular frame groups")
        print(f"  - Generated {total_multiview_groups} multi-view frame groups")
        print(f"  - Total: {total_groups} frame groups")
    
    return frame_groups


def _generate_monocular_groups(
    person_camera_frames: Dict[str, List[int]], 
    frames_per_sample: int,
    max_input_frames: int = 16,
    samples_multiplier: int = 5
) -> List[List[Dict[str, Union[str, int]]]]:
    """
    Generate monocular groups: each group contains frames from the same camera.
    
    Args:
        person_camera_frames: Dict mapping camera_id to list of frame indices
        frames_per_sample: Number of frames per sample
        max_input_frames: Maximum number of input frames
        samples_multiplier: Multiplier for number of samples (default: 5)
        
    Returns:
        List of groups, each group is a list of camera-frame pairs with input_frames info
    """
    groups = []
    
    for camera_id, frames in person_camera_frames.items():
        if len(frames) < frames_per_sample:
            continue
            
        # Generate groups by sampling frames from this camera
        frames_array = np.array(frames)
        num_groups = len(frames_array) // frames_per_sample
        
        # Multiply the number of groups by samples_multiplier
        total_samples = num_groups * samples_multiplier
        
        for _ in range(total_samples):
            # Sample frames_per_sample frames without replacement
            selected_indices = np.random.choice(
                len(frames_array), 
                size=frames_per_sample, 
                replace=False
            )
            selected_frames = frames_array[selected_indices]
            
            # Randomly determine input_frames for this group (2 to max_input_frames)
            input_frames = np.random.randint(2, max_input_frames + 1)
            
            # Create camera-frame pairs with input_frames info
            group_data = []
            for frame in selected_frames:
                group_data.append({"camera": camera_id, "frame": int(frame)})
            
            # Add input_frames to the group metadata
            group_with_metadata = {
                "data": group_data,
                "input_frames": input_frames
            }
            
            groups.append(group_with_metadata)
    
    return groups


def _generate_multiview_groups(
    person_camera_frames: Dict[str, List[int]], 
    frames_per_sample: int,
    max_input_frames: int = 16,
    samples_multiplier: int = 5
) -> List[List[Dict[str, Union[str, int]]]]:
    """
    Generate multi-view groups: each group contains frames from different cameras.
    
    Args:
        person_camera_frames: Dict mapping camera_id to list of frame indices
        frames_per_sample: Number of camera-frame pairs per sample
        max_input_frames: Maximum number of input frames
        samples_multiplier: Multiplier for number of samples (default: 5)
        
    Returns:
        List of groups, each group is a list of camera-frame pairs with input_frames info
    """
    groups = []
    
    # Create all possible camera-frame pairs
    camera_frame_pairs = []
    for camera_id, frames in person_camera_frames.items():
        for frame in frames:
            camera_frame_pairs.append({"camera": camera_id, "frame": frame})
    
    if len(camera_frame_pairs) < frames_per_sample:
        return groups
    
    # Generate groups by sampling camera-frame pairs
    camera_frame_pairs = np.array(camera_frame_pairs)
    num_groups = len(camera_frame_pairs) // frames_per_sample
    
    # Multiply the number of groups by samples_multiplier
    total_samples = num_groups * samples_multiplier
    
    for _ in range(total_samples):
        # Sample frames_per_sample pairs without replacement
        selected_indices = np.random.choice(
            len(camera_frame_pairs), 
            size=frames_per_sample, 
            replace=False
        )
        selected_pairs = camera_frame_pairs[selected_indices].tolist()
        
        # Randomly determine input_frames for this group (2 to max_input_frames)
        input_frames = np.random.randint(2, max_input_frames + 1)
        
        # Add input_frames to the group metadata
        group_with_metadata = {
            "data": selected_pairs,
            "input_frames": input_frames
        }
        
        groups.append(group_with_metadata)
    
    return groups


def save_frame_groups(frame_groups: Dict[str, Dict[str, List[Dict[str, Union[str, int]]]]], 
                     meta_path: str, task_type: str = "monocular") -> None:
    """
    Save multi-view frame groups to JSON file.
    
    Args:
        frame_groups: Dictionary containing multi-view frame groups
        meta_path: Path to save the JSON file
        task_type: Type of task for display purposes
    """
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    
    with open(meta_path, 'w') as f:
        f.write('{\n')
        sorted_items = sorted(frame_groups.items())
        for i, (key, data) in enumerate(sorted_items):
            # Format data array
            data_parts = []
            for item in data['data']:
                camera = item['camera']
                frame = item['frame']
                data_parts.append(f'{{"camera":"{camera}","frame":{frame}}}')
            
            data_str = '[' + ','.join(data_parts) + ']'
            
            # Add input_frames if it exists
            if 'input_frames' in data:
                input_frames = data['input_frames']
                full_data_str = f'{{"data":{data_str},"input_frames":{input_frames}}}'
            else:
                full_data_str = f'{{"data":{data_str}}}'
            
            f.write(f'    "{key}": {full_data_str}')
            if i < len(sorted_items) - 1:
                f.write(',')
            f.write('\n')
        f.write('}\n')
    
    # Display appropriate message based on task type
    if task_type == "monocular":
        print(f"Saved {len(frame_groups)} monocular frame groups to {meta_path}")
    elif task_type == "multi-view":
        print(f"Saved {len(frame_groups)} multi-view frame groups to {meta_path}")
    elif task_type == "unified":
        print(f"Saved {len(frame_groups)} unified frame groups to {meta_path}")
    else:
        print(f"Saved {len(frame_groups)} frame groups to {meta_path}")


def generate_dataset_json(
    root_dir: str,
    meta_path: str,
    frames_per_sample: int = 32,
    seed: int = 42,
    task_type: str = "monocular",
    max_input_frames: int = 16,
    samples_multiplier: int = 5,
    force_regenerate: bool = False
) -> bool:
    """
    Generate dataset JSON file from root directory.
    
    Args:
        root_dir: Root directory containing the dataset
        meta_path: Path to save the JSON file
        frames_per_sample: Number of frames per sample
        seed: Random seed for frame selection
        task_type: Type of task - "monocular", "multi-view", or "unified"
        max_input_frames: Maximum number of input frames (default: 16)
        force_regenerate: Force regeneration even if file exists
    
    Returns:
        True if file was generated or already exists, False otherwise
    """
    # Check if meta file exists and is not empty
    if not force_regenerate and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                frame_groups = json.load(f)
                if len(frame_groups) > 0:
                    print(f"Dataset JSON file already exists at {meta_path} ({len(frame_groups)} groups)")
                    return True
        except json.JSONDecodeError:
            print(f"Warning: {meta_path} is not a valid JSON file. Will regenerate.")
    
    # Generate meta file
    print(f"Generating dataset JSON file at {meta_path}")
    frame_groups = process_root_directory(root_dir, frames_per_sample, seed, task_type, max_input_frames, samples_multiplier)
    
    if len(frame_groups) == 0:
        print("No valid frame groups found!")
        return False
    
    save_frame_groups(frame_groups, meta_path, task_type)
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate dataset JSON file for VGGTAvatar')
    parser.add_argument('--root_dir', type=str, default="/Data/wuyue/nersemble_FLAME",
                       help='Root directory containing the dataset (default: /Data/wuyue/nersemble_FLAME)')
    parser.add_argument('--meta_path', type=str, default="./datasets/nersemble_uids.json",
                       help='Path to save the JSON file (default: ./datasets/nersemble_uids.json)')
    parser.add_argument('--frames_per_sample', type=int, default=32,
                       help='Number of frames per sample (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for frame selection (default: 42)')
    parser.add_argument('--task_type', type=str, default="monocular",
                       choices=["monocular", "multi-view", "unified"],
                       help='Task type: monocular, multi-view, or unified (default: monocular)')
    parser.add_argument('--max_input_frames', type=int, default=16,
                       help='Maximum number of input frames (default: 16)')
    parser.add_argument('--samples_multiplier', type=int, default=5,
                       help='Multiplier for number of samples (default: 5)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if file exists')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory {args.root_dir} does not exist!")
        return 1
    
    if args.frames_per_sample <= 0:
        print(f"Error: frames_per_sample must be positive, got {args.frames_per_sample}")
        return 1
    
    # Generate dataset JSON
    success = generate_dataset_json(
        root_dir=args.root_dir,
        meta_path=args.meta_path,
        frames_per_sample=args.frames_per_sample,
        seed=args.seed,
        task_type=args.task_type,
        max_input_frames=args.max_input_frames,
        samples_multiplier=args.samples_multiplier,
        force_regenerate=args.force
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 