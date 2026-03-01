#!/usr/bin/env python3
"""
Dataset preprocessing utility for VGGTAvatar.
This script generates dataset JSON files from raw data directories.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


class JsonStreamWriter:
    """
    Helper class for streaming JSON writing.
    Writes items incrementally to a JSON object structure: 
    {
        "key1": value1,
        "key2": value2,
        ...
    }
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.f = None
        self.first_item = True
        self.count = 0

    def __enter__(self):
        dirname = os.path.dirname(self.filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.f = open(self.filename, 'w')
        self.f.write('{\n')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f:
            self.f.write('\n}\n')
            self.f.close()

    def add_item(self, key: str, value: any):
        """Write a single key-value pair."""
        if not self.first_item:
            self.f.write(',\n')
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        json_str = json.dumps(value, cls=NumpyEncoder)
        self.f.write(f'  "{key}": {json_str}')
        
        self.first_item = False
        self.count += 1
        
        # Optional: Flush periodically
        if self.count % 100 == 0:
            self.f.flush()

    def get_count(self):
        return self.count


def collect_valid_frames(processed_data_root: str) -> List[int]:
    """Collect valid frame indices from processed_data directory."""
    required_files = {'rgb.npy', 'mask.npy', 'intrs.npy', 'landmark2d.npz', 'bg_color.npy'}

    try:
        if not os.path.exists(processed_data_root):
            return []
            
        all_entries = os.listdir(processed_data_root)
        frame_dirs = [d for d in all_entries
                    if d.isdigit() and os.path.isdir(os.path.join(processed_data_root, d))]
        frame_dirs.sort(key=int)

        required_set = set(required_files)
        camera_frames = []
        for d in frame_dirs:
            frame_idx = int(d)
            frame_path = os.path.join(processed_data_root, d)
            try:
                frame_files = os.listdir(frame_path)
                if required_set.issubset(set(frame_files)):
                    camera_frames.append(frame_idx)
            except OSError:
                continue
        return camera_frames
    except OSError:
        return []


def _generate_groups_by_strategy(
    person_camera_frames: Dict[str, any],
    task_type: str,
    max_input_frames: int = 16,
    target_frames: int = 16,
    rng: Optional[np.random.Generator] = None,
    multiply: int = 1
) -> List[Dict[str, any]]:
    """
    Generate groups based on dataset strategy.
    If multiply > 1, generate multiple versions with different random seeds.
    """
    all_groups = []

    for i in range(multiply):
        # Create a new RNG for each multiplication iteration
        current_seed = i if rng is None else rng.integers(0, 2**32) + i
        current_rng = np.random.default_rng(current_seed)

        if task_type == "monocular":
            groups = _generate_monocular_groups(person_camera_frames, max_input_frames, target_frames, current_rng)
        else:
            # Default to multiview (mixed cameras)
            groups = _generate_multiview_groups(person_camera_frames, max_input_frames, target_frames, current_rng)

        all_groups.extend(groups)

    return all_groups



def _generate_monocular_groups(
    person_camera_frames: Dict[str, any],
    max_input_frames: int = 16,
    target_frames: int = 16,
    rng: Optional[np.random.Generator] = None
) -> List[Dict[str, any]]:
    """Generates groups for monocular data (single camera)."""
    groups = []
    
    # In monocular mode, we expect person_camera_frames to contain ONE camera
    if not person_camera_frames:
        return groups

    # Use provided RNG or create default
    if rng is None:
        rng = np.random.default_rng()

    # Use the first available camera
    camera_id = list(person_camera_frames.keys())[0]
    camera_data = person_camera_frames[camera_id]

    if isinstance(camera_data, dict):
        frames = camera_data['frames']
        sequence = camera_data['sequence']
    else:
        frames = camera_data
        sequence = None

    total_frames = len(frames)
    if total_frames < target_frames + 1:
        return groups

    # Define input frames schedule
    input_frame_counts = list(range(2, 16))
    base_schedule = []
    current = 16
    while current <= max_input_frames:
        base_schedule.append(current)
        current *= 2
    input_frames_schedule = input_frame_counts + base_schedule

    # Shuffle frames to create diversity
    shuffled_frames = list(frames)
    rng.shuffle(shuffled_frames)

    max_possible_input = total_frames - target_frames
    valid_input_frames = [n for n in input_frames_schedule if n <= max_possible_input]

    for input_frames in valid_input_frames:
        # Strategy: Pick input_frames random inputs, and 16 random targets (but from the same set)
        # Note: The original logic took first N as input and last M as target from the SHUFFLED list.
        # This effectively means disjoint sets if total > input + target.
        
        input_frame_nums = shuffled_frames[:input_frames]
        target_frame_nums = shuffled_frames[-target_frames:]

        data = []
        for frame in input_frame_nums + target_frame_nums:
            data.append({
                "camera": camera_id,
                "frame": frame,
                "seq": sequence
            })

        groups.append({
            "data": data,
            "input_frames": input_frames
        })

    return groups




def process_dataset_monocular(
    root_dir: str,
    output_writer: JsonStreamWriter,
    max_input_frames: int = 16,
    target_frames: int = 16,
    seed: int = 42,
    dataset_name: str = "",
    multiply: int = 1
):
    """
    Process dataset in Monocular mode.
    Strictly seperates sequences: person/camera/sequence is an independent unit.
    """
    print(f"Processing MONOCULAR mode: {root_dir} (dataset: {dataset_name})")
    print(f"Structure expected: {root_dir}/<person>/<camera>/<sequence>/...")

    if not os.path.isdir(root_dir):
        print(f"Root dir {root_dir} does not exist.")
        return

    # List persons
    persons = sorted([p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))])
    
    # Initialize RNG once
    rng = np.random.default_rng(seed)
    
    global_group_counter = 0

    for person_id in tqdm(persons, desc=f"Persons ({dataset_name})"):
        person_dir = os.path.join(root_dir, person_id)
        
        cameras = sorted([c for c in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, c))])
        for camera_id in cameras:
            camera_dir = os.path.join(person_dir, camera_id)
            
            seqs = sorted([s for s in os.listdir(camera_dir) if os.path.isdir(os.path.join(camera_dir, s))])
            for seq_name in seqs:
                if "head" in seq_name.lower(): continue # Skip generic head meshes if any
                
                seq_dir = os.path.join(camera_dir, seq_name)
                processed_dir = os.path.join(seq_dir, "processed_data")
                
                frames = collect_valid_frames(processed_dir)
                if not frames: continue

                # Prepare data structure for generator
                person_camera_frames = {
                    camera_id: {
                        "frames": frames,
                        "sequence": seq_name
                    }
                }
                
                # Generate
                groups = _generate_groups_by_strategy(
                    person_camera_frames,
                    "monocular",
                    max_input_frames,
                    target_frames,
                    rng,
                    multiply
                )

                for grp in groups:
                    global_group_counter += 1
                    
                    # Prefix key with dataset_name if provided
                    unique_id = f"{person_id}/{global_group_counter:07d}"
                    if dataset_name:
                        output_key = f"{dataset_name}/{unique_id}"
                    else:
                        output_key = unique_id
                        
                    output_writer.add_item(output_key, grp)


def process_dataset_unified(
    root_dir: str,
    output_writer: JsonStreamWriter,
    max_input_frames: int = 16,
    target_frames: int = 16,
    seed: int = 42,
    dataset_name: str = "",
    multiply: int = 1
):
    """
    Process dataset in Unified mode.
    Aggregates ALL cameras and sequences for a single Person.
    """
    print(f"Processing UNIFIED mode: {root_dir} (dataset: {dataset_name})")
    
    if not os.path.isdir(root_dir):
        return

    persons = sorted([p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))])

    # Initialize RNG once
    rng = np.random.default_rng(seed)

    for person_id in tqdm(persons, desc=f"Persons ({dataset_name})"):
        person_dir = os.path.join(root_dir, person_id)
        
        # Collect ALL frames for this person
        person_camera_frames = {} 
        
        cameras = sorted([c for c in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, c))])
        for camera_id in cameras:
            camera_dir = os.path.join(person_dir, camera_id)
            seqs = sorted([s for s in os.listdir(camera_dir) if os.path.isdir(os.path.join(camera_dir, s))])
            
            for seq_name in seqs:
                if "head" in seq_name.lower(): continue
                
                seq_dir = os.path.join(camera_dir, seq_name)
                processed_dir = os.path.join(seq_dir, "processed_data")
                frames = collect_valid_frames(processed_dir)
                if not frames: continue
                
                # Create a unique key for this batch of frames so they accumulate
                unique_key = f"{camera_id}::{seq_name}"
                person_camera_frames[unique_key] = {
                    "frames": frames,
                    "sequence": seq_name,
                }
        
        if not person_camera_frames:
            continue

        groups = _generate_groups_by_strategy(
            person_camera_frames,
            "multi-view",
            max_input_frames,
            target_frames,
            rng,
            multiply
        )
        
        for grp in groups:
            # Prefix key with dataset_name if provided
            unique_id = f"{person_id}/{output_writer.get_count() + 1:07d}"
            if dataset_name:
                output_key = f"{dataset_name}/{unique_id}"
            else:
                output_key = unique_id
                
            output_writer.add_item(output_key, grp)

        # Also generate monocular samples for each camera-sequence pair
        for unique_key, data in person_camera_frames.items():
            if "::" in unique_key:
                real_cam_id = unique_key.split("::")[0]
            else:
                real_cam_id = unique_key
            
            # Construct single camera dict
            single_cam_data = {
                real_cam_id: data
            }
            
            mono_groups = _generate_groups_by_strategy(
                single_cam_data,
                "monocular",
                max_input_frames,
                target_frames,
                rng,
                multiply
            )
            
            for grp in mono_groups:
                # Prefix key with dataset_name if provided
                unique_id = f"{person_id}/{output_writer.get_count() + 1:07d}"
                if dataset_name:
                    output_key = f"{dataset_name}/{unique_id}"
                else:
                    output_key = unique_id
                    
                output_writer.add_item(output_key, grp)


# Re-implementing _generate_multiview_groups to handle the key issue
def _generate_multiview_groups(
    person_camera_frames: Dict[str, any],
    max_input_frames: int = 16,
    target_frames: int = 16,
    rng: Optional[np.random.Generator] = None
) -> List[Dict[str, any]]:
    """Generates groups for multi-view data (mixing cameras/sequences)."""
    groups = []
    if rng is None:
        rng = np.random.default_rng()

    # Collect ALL frames from ALL cameras/sequences
    all_pairs = []
    for key, data in person_camera_frames.items():
        if isinstance(data, dict):
            frames = data['frames']
            seq = data['sequence']
            # Parsing real camera ID if encoded in key "cam::seq"
            if "::" in key:
                real_cam = key.split("::")[0]
            else:
                real_cam = key
            
            for f in frames:
                all_pairs.append({"camera": real_cam, "frame": f, "seq": seq})
        else:
            # Fallback
            for f in data:
                all_pairs.append({"camera": key, "frame": f})

    total_frames = len(all_pairs)
    if total_frames < target_frames + 1:
        return groups

    # Input schedule
    input_frame_counts = list(range(2, 16))
    base_schedule = []
    current = 16
    while current <= max_input_frames:
        base_schedule.append(current)
        current *= 2
    input_frames_schedule = input_frame_counts + base_schedule

    # Chunking strategy
    max_sched_input = max(input_frames_schedule) if input_frames_schedule else 16
    ideal_chunk_size = max_sched_input + target_frames
    
    if total_frames < ideal_chunk_size:
        available_input = total_frames - target_frames
        if available_input <= 0: return groups
        chunk_size = available_input + target_frames
    else:
        chunk_size = ideal_chunk_size

    num_chunks = max(1, total_frames // chunk_size)
    rng.shuffle(all_pairs)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_data = all_pairs[start:end]
        
        available = len(chunk_data)
        current_max_input = available - target_frames
        if current_max_input <= 0: continue
        
        valid_input_counts = [n for n in input_frames_schedule if n <= current_max_input]
        
        for cnt in valid_input_counts:
            inputs = chunk_data[:cnt]
            targets = chunk_data[-target_frames:]
            groups.append({
                "data": inputs + targets,
                "input_frames": cnt
            })

    return groups
