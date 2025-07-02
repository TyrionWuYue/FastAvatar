# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
import traceback
import json
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, List, Dict
from megfile import smart_open, smart_path_join, smart_exists
import random
import os


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dirs: str, meta_path: Optional[Union[list, str]]):
        super().__init__()
        self.root_dirs = root_dirs
        self.uids = self._load_uids(meta_path)

    def __len__(self):
        return len(self.uids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            # raise e
            return self.__getitem__((idx + 1) % self.__len__())

    @staticmethod
    def _load_uids(meta_path: Optional[Union[list, str]]):
        # meta_path is a json file
        if isinstance(meta_path, str):
            with open(meta_path, 'r') as f:
                uids = json.load(f)
        else:
            uids_lst = []
            max_total = 0
            for pth, weight in meta_path:
                with open(pth, 'r') as f:
                    uids = json.load(f)
                    max_total = max(len(uids) / weight, max_total)
                uids_lst.append([uids, weight, pth])
            merged_uids = []
            for uids, weight, pth in uids_lst:
                repeat = 1
                if len(uids) < int(weight * max_total):
                    repeat = int(weight * max_total) // len(uids)
                cur_uids = uids * repeat
                merged_uids += cur_uids
                print("Data Path:", pth, "Repeat:", repeat, "Final Length:", len(cur_uids))
            uids = merged_uids
            print("Total UIDs:", len(uids))
        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0):
        ''' Load and blend RGBA image to RGB with certain background, 0-1 scaled '''
        rgba = np.array(Image.open(smart_open(file_path, 'rb')))
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])
        rgba[:, :3, ...] * rgba[:, 3:, ...] + (1 - rgba[:, 3:, ...])
        return rgb

    @staticmethod
    def _locate_datadir(root_dirs, uid, locator: str):
        for root_dir in root_dirs:
            datadir = smart_path_join(root_dir, uid, locator)
            if smart_exists(datadir):
                return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for uid {uid}")


class FrameBaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, 
                 root_dir: str, 
                 meta_path: Optional[Union[list, str]],
                 input_frames: int = 5,
                 frames_per_sample: int = 10):
        super().__init__()
        self.root_dir = root_dir
        self.meta_path = meta_path
        self.input_frames = input_frames
        self.frames_per_sample = frames_per_sample
        self.uids = self._load_uids(self)

        assert self.frames_per_sample > self.input_frames, "frames_per_sample must be greater than input_frames"
    
    def __len__(self):
        return len(self.uids)
    
    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            # raise e
            return self.__getitem__((idx + 1) % self.__len__())
    
    @staticmethod
    def _load_root_dir(self) -> None:
        """Process the root directory and generate frame groups.
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
                        # (no longer: 00000/processed_data/)
        """
        frame_groups: Dict[str, Dict[str, Union[str, List[int]]]] = {}
        for sequence_name in sorted(os.listdir(self.root_dir)):
            sequence_dir = os.path.join(self.root_dir, sequence_name)
            if not os.path.isdir(sequence_dir):
                continue
            for person_id in sorted(os.listdir(sequence_dir)):
                person_dir = os.path.join(sequence_dir, person_id)
                if not os.path.isdir(person_dir):
                    continue
                for camera_id in sorted(os.listdir(person_dir)):
                    camera_dir = os.path.join(person_dir, camera_id)
                    if not os.path.isdir(camera_dir):
                        continue
                    processed_data_root = os.path.join(camera_dir, "processed_data")
                    if not os.path.isdir(processed_data_root):
                        print(f"[SKIP] {processed_data_root} does not exist.")
                        continue
                    camera_frames = []
                    for d in sorted(os.listdir(processed_data_root)):
                        frame_path = os.path.join(processed_data_root, d)
                        if os.path.isdir(frame_path) and d.isdigit():
                            required_files = ['rgb.npy', 'mask.npy', 'intrs.npy', 'landmark2d.npz', 'bg_color.npy']
                            missing_files = [f for f in required_files if not os.path.exists(os.path.join(frame_path, f))]
                            if missing_files:
                                print(f"[SKIP] {frame_path} missing files: {missing_files}")
                                continue
                            camera_frames.append(int(d))
                    if len(camera_frames) < self.frames_per_sample:
                        print(f"[SKIP] Camera {camera_id} in {sequence_name}/{person_id}/{camera_id} has only {len(camera_frames)} valid frames, need {self.frames_per_sample}")
                        continue
                    groups = []
                    remaining_frames = camera_frames.copy()
                    while len(remaining_frames) >= self.frames_per_sample:
                        selected_indices = np.random.choice(
                            len(remaining_frames),
                            size=self.frames_per_sample,
                            replace=False
                        )
                        selected_frames = [remaining_frames[i] for i in selected_indices]
                        groups.append(selected_frames)
                        remaining_frames = [f for i, f in enumerate(remaining_frames) if i not in selected_indices]
                    if remaining_frames:
                        additional_needed = self.frames_per_sample - len(remaining_frames)
                        if additional_needed > 0:
                            all_used_frames = [f for group in groups for f in group]
                            additional_frames = np.random.choice(
                                all_used_frames,
                                size=additional_needed,
                                replace=False
                            )
                            remaining_frames.extend(additional_frames)
                        groups.append(remaining_frames)
                    base_key = f"{sequence_name}/{person_id}/{camera_id}"
                    for group_idx, group in enumerate(groups, 1):
                        key = f"{base_key}/{group_idx:05d}"
                        frame_groups[key] = {
                            "camera": camera_id,
                            "frames": group
                        }
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, 'w') as f:
            f.write('{')
            sorted_items = sorted(frame_groups.items())
            for i, (key, data) in enumerate(sorted_items):
                frames_str = '[' + ','.join(str(f) for f in data['frames']) + ']'
                data_str = f'{{"camera":"{data["camera"]}","frames":{frames_str}}}'
                f.write(f'    "{key}": {data_str}')
                if i < len(sorted_items) - 1:
                    f.write(',')
                f.write('\n')
            f.write('}')
        print(f"Saved frame groups to {self.meta_path}")
        print(f"Total groups: {len(frame_groups)}")

    @staticmethod
    def _load_uids(self):
        """Load or generate UIDs from meta file.
        
        Args:
            meta_path: Path to the meta json file. If file doesn't exist or is empty,
                     it will be generated using _load_root_dir.
        
        Returns:
            list: A list of UIDs, where each UID is a tuple of (sequence_path, frame_data)
                 sequence_path includes a unique group ID (e.g., "sequence/person/camera/00001")
                 frame_data is a dict with "camera" and "frames" keys
        """
        # Check if meta file exists and is not empty
        need_generate = True
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                try:
                    frame_groups = json.load(f)
                    if len(frame_groups) > 0:
                        need_generate = False
                except json.JSONDecodeError:
                    print(f"Warning: {self.meta_path} is not a valid JSON file. Will regenerate.")
        
        # Generate meta file if needed
        if need_generate:
            print(f"Generating meta file at {self.meta_path}")
            self._load_root_dir(self)
            # Read the newly generated file
            with open(self.meta_path, 'r') as f:
                frame_groups = json.load(f)
        
        # Convert frame_groups to uids list
        # Each UID will be a tuple of (sequence_path_with_group_id, frame_data)
        uids = [(path, frame_data) for path, frame_data in frame_groups.items()]
        
        print(f"Loaded {len(uids)} frame groups")
        return uids



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FrameBasedDataset')
    parser.add_argument('--root_dir', type=str, default="/mnt/Data/wuyue/nersemble_FLAME", help='Root directory containing both image and FLAME folders')
    parser.add_argument('--meta_path', type=str, default="/home/tjwr/wuyue/Avatar/VGGTAvatar/datasets/nersemble_uids.json", help='Path to meta json file')
    parser.add_argument('--frames_per_sample', type=int, default=20, help='Number of frames per sample')
    args = parser.parse_args()

    # Create test dataset
    dataset = FrameBaseDataset(
        root_dir=args.root_dir,
        meta_path=args.meta_path,
        frames_per_sample=args.frames_per_sample,
    )