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
    def _load_uids(self):
        """Load UIDs from meta file.
        
        Returns:
            list: A list of UIDs, where each UID is a tuple of (sequence_path, frame_data)
                 For the new structure, frame_data is a dict with "data" key containing 
                 a list of camera-frame pairs
        """
        # Check if meta file exists
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"Meta file {self.meta_path} not found. "
                f"Please run preprocessing first: python scripts/preprocess_before_train.py --config your_config.yaml"
            )
        
        # Load frame groups from JSON file
        with open(self.meta_path, 'r') as f:
            try:
                frame_groups = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON file: {self.meta_path}")
        
        if len(frame_groups) == 0:
            raise ValueError(f"Meta file {self.meta_path} is empty")
        
        # Convert frame_groups to uids list
        uids = []
        for path, frame_data in frame_groups.items():
            # Check if this is the new structure (with "data" field)
            if "data" in frame_data:
                # New structure: {"data": [{"camera": "cam1", "frame": 1}, ...]}
                uids.append((path, frame_data))
            else:
                # Legacy structure: {"camera": "cam1", "frames": [1, 2, 3, ...]}
                # Convert to new structure for backward compatibility
                converted_data = {"data": []}
                for frame in frame_data["frames"]:
                    converted_data["data"].append({
                        "camera": frame_data["camera"],
                        "frame": frame
                    })
                uids.append((path, converted_data))
        
        print(f"Loaded {len(uids)} frame groups from {self.meta_path}")
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