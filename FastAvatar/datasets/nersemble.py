from collections import defaultdict
import os
import numpy as np
import torch
import json
import random

from FastAvatar.datasets.base import FrameBaseDataset
from FastAvatar.utils.proxy import no_proxy
from typing import Optional, Union

__all__ = ['NersembleDataset']


class NersembleDataset(FrameBaseDataset):

    """
    Structure of the dataset:
        'rgbs': [N, 3, H, W]
        'frame_indices': [N]
        'valid_mask': [N]
        'camera_params': {
            'cx': [N],
            'cy': [N],
            'fl_x': [N],
            'fl_y': [N],
            'h': [N],
            'w': [N],
            'transform_matrix': [N, 4, 4]
        }
        'rotation': [N, 3]
        'neck_pose': [N, 3]
        'jaw_pose': [N, 3]
        'eyes_pose': [N, 6]
        'translation': [N, 3]
        'shape': [N, 300]  # Identity parameters - same values repeated N times
        'expr': [N, 100]  # Expression parameters - can vary per frame
    """

    def __init__(self, root_dirs: str,
                 meta_path: Optional[Union[list, str]],
                 input_frames: int,
                 frames_per_sample: int,
                 render_image_res: int,
                 source_image_res: int,
                 disorder=True,
                 aspect_standard=1.0,  # h/w
                 is_val=False,
                 val_num=64,
                 val_id=None,
                 use_teeth=False,
                 **kwargs):
        
        """
        Args:
            root_dirs: Root directory containing sequence folders
            frames_per_sample: Number of frames to include in each sample
            render_image_res: Target resolution for rendered images
            source_image_res: Source image resolution
            disorder: Whether to randomly sample frames or use sequential frames
            sequence_list_path: Path to json file containing list of sequences to use
            val_id: List of IDs for validation. If None or empty, randomly select 5 IDs
            val_num: Number of samples to select from validation IDs
        """
        super().__init__(
            root_dir=root_dirs,
            input_frames=input_frames,
            frames_per_sample=frames_per_sample,
            meta_path=meta_path
        )
        self.max_input_frames = input_frames
        self.target_frames = self.frames_per_sample - self.max_input_frames
        self.render_image_res = render_image_res
        self.source_image_res = source_image_res
        self.aspect_standard = aspect_standard
        self.disorder = disorder
        self.is_val = is_val
        self.use_random_input = kwargs.get("use_random_input", True)  # Add switch for random input frames
        self.use_teeth = use_teeth
        
        # Split train/val based on val_id
        self._split_train_val(val_num, val_id)
    
    @staticmethod
    def _extract_id_from_path(path: str) -> str:
        """Extract ID from path. For nersemble, ID is the first part of path."""
        # path format: "sequence_name/person_id" or "sequence_name"
        parts = path.split('/')
        return parts[0] if len(parts) > 0 else path
    
    def _split_train_val(self, val_num: int, val_id: Optional[list]):
        """Split dataset into train/val based on val_id list."""
        if val_id is None:
            val_id = []
        
        # Get all unique IDs from uids
        all_ids = set()
        id_to_uids = {}
        for idx, (path, frame_data) in enumerate(self.uids):
            uid_id = self._extract_id_from_path(path)
            all_ids.add(uid_id)
            if uid_id not in id_to_uids:
                id_to_uids[uid_id] = []
            id_to_uids[uid_id].append(idx)
        
        # Determine which IDs to use for validation
        if len(val_id) == 0:
            # Randomly select 5 IDs if val_id is empty
            # Use fixed seed to ensure consistency across all processes/GPUs
            available_ids = sorted(list(all_ids))  # Sort for deterministic order
            if len(available_ids) < 5:
                selected_ids = available_ids
            else:
                # Use fixed seed for reproducible validation set selection
                rng = random.Random(42)  # Fixed seed
                selected_ids = rng.sample(available_ids, 5)
            print(f"No val_id specified, randomly selected {len(selected_ids)} IDs for validation (with fixed seed): {selected_ids}")
        else:
            # Use specified val_id, filter out non-existent ones
            selected_ids = [uid for uid in val_id if uid in all_ids]
            missing_ids = [uid for uid in val_id if uid not in all_ids]
            if missing_ids:
                print(f"Warning: Some val_id not found in dataset: {missing_ids}")
            if len(selected_ids) == 0:
                print(f"Warning: No valid val_id found, using all IDs")
                selected_ids = list(all_ids)
            else:
                print(f"Using {len(selected_ids)} specified IDs for validation: {selected_ids}")
        
        # Collect all UID indices for validation IDs
        val_uid_indices = []
        for uid_id in selected_ids:
            val_uid_indices.extend(id_to_uids[uid_id])
        
        # Sample val_num samples from validation UIDs
        if self.is_val:
            # Categorize samples by camera type (multi-view vs monocular)
            multi_view_indices = []
            monocular_indices = []
            
            for idx in val_uid_indices:
                path, frame_data = self.uids[idx]
                cameras = set([pair['camera'] for pair in frame_data['data']])
                if len(cameras) > 1:
                    multi_view_indices.append(idx)
                else:
                    monocular_indices.append(idx)
            
            # Sample balanced validation set
            half_val = val_num // 2
            
            if len(multi_view_indices) >= half_val:
                selected_multi = random.sample(multi_view_indices, half_val)
            else:
                selected_multi = multi_view_indices
                print(f"Warning: Only {len(multi_view_indices)} multi-view samples available, wanted {half_val}")
            
            if len(monocular_indices) >= half_val:
                selected_mono = random.sample(monocular_indices, half_val)
            else:
                selected_mono = monocular_indices
                print(f"Warning: Only {len(monocular_indices)} monocular samples available, wanted {half_val}")
            
            val_uid_indices = selected_multi + selected_mono
            self.uids = [self.uids[i] for i in val_uid_indices]
            
            print(f"Validation set: {len(selected_multi)} multi-view + {len(selected_mono)} monocular = {len(self.uids)} total samples from {len(selected_ids)} IDs")
        else:
            # Training set: exclude validation IDs
            train_uid_indices = [i for i in range(len(self.uids)) if i not in val_uid_indices]
            self.uids = [self.uids[i] for i in train_uid_indices]
            print(f"Training set: {len(self.uids)} samples (excluded {len(selected_ids)} validation IDs)")

    @staticmethod
    def _load_pose(frame_info):
        c2w = torch.eye(4)
        c2w = np.array(frame_info["transform_matrix"])
        c2w[:3, 1:3] *= -1
        c2w = torch.FloatTensor(c2w)
        
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = frame_info["fl_x"]
        intrinsic[1, 1] = frame_info["fl_y"]
        intrinsic[0, 2] = frame_info["cx"]
        intrinsic[1, 2] = frame_info["cy"]
        intrinsic = intrinsic.float()
        
        return c2w, intrinsic
    
    def load_flame_params(self, flame_file_path, frame_idx, teeth_bs=None):
        flame_param = dict(np.load(flame_file_path, allow_pickle=True))
        flame_param_tensor = {}
        # Use frame_idx to index per-frame params if they are arrays with more than 1 entry
        for k in ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']:
            v = flame_param[k]
            if v.shape[0] > 1:
                flame_param_tensor[k] = torch.FloatTensor(v[frame_idx])
            else:
                flame_param_tensor[k] = torch.FloatTensor(v[0])
        if teeth_bs is not None:
            flame_param_tensor['teeth_bs'] = torch.FloatTensor(teeth_bs)
        flame_param_tensor['shape'] = torch.FloatTensor(flame_param['shape'])
        return flame_param_tensor
    
    def get_frame_info(self, path: str, frame_idx: int, camera_id: str, sequence_name: str = None):
        """Get frame info from transforms.json file at the camera directory level.
        Args:
            path: Path to the sequence/person directory
            frame_idx: Frame index
            camera_id: Camera ID
            sequence_name: Sequence Name (optional)
        Returns:
            frame_info: Complete frame information from transforms.json
        """
        if sequence_name:
            camera_dir = os.path.join(self.root_dir, path, camera_id, sequence_name)
        else:
            camera_dir = os.path.join(self.root_dir, path, camera_id)
        
        transforms_json = os.path.join(camera_dir, "transforms.json")
        if not os.path.exists(transforms_json):
            raise FileNotFoundError(f"transforms.json not found at {transforms_json}")
        with open(transforms_json) as fp:
            data = json.load(fp)
        frame_info = None
        for f in data['frames']:
            if str(frame_idx).zfill(5) in f.get('file_path', ''):
                frame_info = f
                break
        if frame_info is None:
            try:
                frame_info = data['frames'][frame_idx]
            except Exception:
                raise ValueError(f"Frame {frame_idx} not found in {transforms_json}")
        frame_info['flame_param_path'] = f"flame_param/{frame_idx:05d}.npz"
        return frame_info

    def load_processed_data(self, frame_dir):
        """Load all processed data from processed_data/<frame_idx>/ directory for a specific frame index."""
        rgb = torch.from_numpy(np.load(os.path.join(frame_dir, 'rgb.npy'))).float()
        mask = torch.from_numpy(np.load(os.path.join(frame_dir, 'mask.npy'))).float()
        intr = torch.from_numpy(np.load(os.path.join(frame_dir, 'intrs.npy'))).float()
        landmarks_data = np.load(os.path.join(frame_dir, 'landmark2d.npz'))
        landmarks = torch.from_numpy(landmarks_data['face_landmark_2d'][0]).float()
        bg_color = np.load(os.path.join(frame_dir, 'bg_color.npy'))
        return rgb, mask, intr, landmarks, bg_color

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]
        path, frame_data = uid
        
        # Extract camera-frame pairs and input_frames from the new data structure
        camera_frame_pairs = frame_data["data"]
        input_frames = frame_data.get("input_frames", self.max_input_frames)  # Use pre-defined input_frames
        
        # Extract base path (remove group_id from path)
        # path format: "person_id/camera_id/sequence_name/group_id"
        path_parts = path.split('/')
        base_path = '/'.join(path_parts[:-1])  # Remove group_id
       
        # Load exactly input_frames + target_frames pairs (not max_input_frames)
        total_frames_needed = input_frames + self.target_frames
        if len(camera_frame_pairs) > total_frames_needed:
            rng = np.random.RandomState(idx)
            start_idx = rng.randint(0, len(camera_frame_pairs) - total_frames_needed + 1)
            camera_frame_pairs = camera_frame_pairs[start_idx:start_idx + total_frames_needed]
        elif len(camera_frame_pairs) < total_frames_needed:
            raise ValueError(f"Not enough camera-frame pairs available. Need {total_frames_needed} pairs but only have {len(camera_frame_pairs)}")

        # Load data for each camera-frame pair
        c2ws, intrs, rgbs, bg_colors, masks, landmarks_list = [], [], [], [], [], []

        for i, pair in enumerate(camera_frame_pairs):
            camera_id = pair["camera"]
            frame_idx = pair["frame"]
            sequence_name = pair.get("seq")
            
            # Get complete frame info from transforms.json
            frame_info = self.get_frame_info(base_path, frame_idx, camera_id, sequence_name=sequence_name)
            
            # Build frame directory path
            # base_path already includes person_id/camera_id/sequence_name
            if sequence_name:
                frame_dir = os.path.join(self.root_dir, base_path, camera_id, sequence_name, "processed_data", f"{frame_idx:05d}")
            else:
                frame_dir = os.path.join(self.root_dir, base_path, "processed_data", f"{frame_idx:05d}")
            
            # Load processed data
            rgb, mask, intrinsic, landmarks, bg_color = self.load_processed_data(frame_dir)
            
            if len(rgb.shape) == 3: 
                rgb = rgb.unsqueeze(0)  # Add batch dimension -> [1, 3, H, W]
            
            if len(mask.shape) == 3: 
                mask = mask.unsqueeze(0)  # Add batch dimension -> [1, 1, H, W]
            
            # Load camera pose using complete frame info
            c2w, _ = self._load_pose(frame_info)
            
            c2ws.append(c2w)
            intrs.append(intrinsic)
            rgbs.append(rgb)
            bg_colors.append(bg_color.item())
            masks.append(mask)
            landmarks_list.append(landmarks)
        
        c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W] 
        bg_colors = torch.tensor(np.array(bg_colors), dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
        landmarks = torch.stack(landmarks_list, dim=0)  # [N, X, 2] 
        
        # Load FLAME parameters for each camera-frame pair
        all_flame_params = defaultdict(list)
        for i, pair in enumerate(camera_frame_pairs):
            camera_id = pair["camera"]
            frame_idx = pair["frame"]
            sequence_name = pair.get("seq")
            
            # Load FLAME parameters
            if sequence_name:
                flame_path = os.path.join(self.root_dir, base_path, camera_id, sequence_name, "flame_param", f"{frame_idx:05d}.npz")
                teeth_bs_pth = os.path.join(self.root_dir, base_path, camera_id, sequence_name, "tracked_teeth_bs.npz")
            else:
                flame_path = os.path.join(self.root_dir, base_path, camera_id, "flame_param", f"{frame_idx:05d}.npz")
                teeth_bs_pth = os.path.join(self.root_dir, base_path, camera_id, "tracked_teeth_bs.npz")
            if os.path.exists(teeth_bs_pth) and self.use_teeth:
                teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
                teeth_bs = teeth_bs_lst[frame_idx] if teeth_bs_lst is not None and frame_idx < len(teeth_bs_lst) else None
            else:
                teeth_bs = None
            
            # Since flame_path is already frame-specific, use index 0 to load the single frame data
            flame_param = self.load_flame_params(flame_path, 0, teeth_bs)
            
            for k, v in flame_param.items():
                all_flame_params[k].append(v)
        
        for k, v in all_flame_params.items():
            all_flame_params[k] = torch.stack(v)  # [N, ...]
        
        # Load canonical flame parameters (use first camera for consistency)
        first_camera_id = camera_frame_pairs[0]["camera"]
        first_seq = camera_frame_pairs[0].get("seq")
        if first_seq:
            canonical_flame_path = os.path.join(self.root_dir, base_path, first_camera_id, first_seq, "canonical_flame_param.npz")
        else:
            canonical_flame_path = os.path.join(self.root_dir, base_path, first_camera_id, "canonical_flame_param.npz")
        canonical_flame_param = self.load_flame_params(canonical_flame_path, 0, None)
        all_flame_params['betas'] = canonical_flame_param['shape'].expand(len(camera_frame_pairs), -1)  # [N, 300]

        ret = {
            'uid': uid,
            'c2ws': c2ws[:input_frames],  # Use actual input_frames instead of max_input_frames
            'target_c2ws': c2ws[-self.target_frames:],
            'intrs': intrs[:input_frames],
            'target_intrs': intrs[-self.target_frames:],
            'rgbs': rgbs[:input_frames],
            'target_rgbs': rgbs[-self.target_frames:],
            'bg_colors': bg_colors[:input_frames],
            'target_bg_colors': bg_colors[-self.target_frames:],
            'masks': masks[:input_frames],
            'target_masks': masks[-self.target_frames:],
            'landmarks': landmarks[:input_frames],
            'target_landmarks': landmarks[-self.target_frames:],
            'max_input_frames': self.max_input_frames,
            'input_frames': input_frames,
            'is_val': self.is_val,
        }
        
        for k, v in all_flame_params.items():
            ret[f'input_{k}'] = v[:input_frames]  # [N_input, ...] - use actual input_frames
            ret[f'target_{k}'] = v[-self.target_frames:]  # [N_target, ...]
        
        # Verify correct tensor shapes
        assert ret['c2ws'].shape[0] == input_frames, f"c2ws shape: {ret['c2ws'].shape}, expected: {input_frames}"
        assert ret['target_c2ws'].shape[0] == self.target_frames
        assert ret['intrs'].shape[0] == input_frames
        assert ret['target_intrs'].shape[0] == self.target_frames
        assert ret['rgbs'].shape[0] == input_frames
        assert ret['target_rgbs'].shape[0] == self.target_frames
        assert ret['bg_colors'].shape[0] == input_frames
        assert ret['masks'].shape[0] == input_frames
        assert ret['target_masks'].shape[0] == self.target_frames
        assert ret['landmarks'].shape[0] == input_frames
        assert ret['target_landmarks'].shape[0] == self.target_frames

        return ret

