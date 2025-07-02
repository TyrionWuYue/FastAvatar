from collections import defaultdict
import os
import numpy as np
import torch
import json
import random

from FastAvatar.datasets.base import FrameBaseDataset
from FastAvatar.utils.proxy import no_proxy
from typing import Optional, Union

__all__ = ['DisorderVideoHeadDataset']


class DisorderVideoHeadDataset(FrameBaseDataset):

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
        'eyes_pose': [N, 3]
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
                 repeat_num=1,
                 aspect_standard=1.0,  # h/w
                 is_val=False,
                 val_num=4,
                 **kwargs):
        
        """
        Args:
            root_dirs: Root directory containing sequence folders
            frames_per_sample: Number of frames to include in each sample
            render_image_res: Target resolution for rendered images
            source_image_res: Source image resolution
            disorder: Whether to randomly sample frames or use sequential frames
            sequence_list_path: Path to json file containing list of sequences to use
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
        self.uids = self.uids * repeat_num
        self.is_val = is_val
        self.use_random_input = kwargs.get("use_random_input", True)  # Add switch for random input frames

        val_indices = random.sample(range(len(self.uids)), val_num)
        if self.is_val:
            self.uids = [self.uids[i] for i in val_indices]
        else:
            self.uids = [self.uids[i] for i in range(len(self.uids)) if i not in val_indices]

    def _get_input_frames(self, idx):
        """Get the number of input frames for a given index.
        This ensures deterministic randomness based on the data index.
        
        With 50% probability:
        - Use max_input_frames
        - Otherwise, randomly select between 1 and max_input_frames
        """
        if self.is_val:
            return self.max_input_frames  # Use fixed input frames during validation
            
        # Use a fixed seed based on the index to ensure reproducibility
        rng = np.random.RandomState(idx)
        
        # 50% chance to use max frames, 50% chance to use random frames
        if rng.random() < 0.5:
            return self.max_input_frames
        else:
            return rng.randint(1, self.max_input_frames + 1)

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
    
    def get_frame_info(self, path: str, frame_idx: int, camera_id: str):
        """Get frame info from transforms.json file at the camera directory level.
        Args:
            path: Path to the sequence/person directory
            frame_idx: Frame index
            camera_id: Camera ID
        Returns:
            frame_info: Complete frame information from transforms.json
        """
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
        
        # Extract camera_id and frames from frame_data
        camera_id = frame_data["camera"]
        frame_indices = frame_data["frames"]
        
        # Extract base path (remove camera_id and group_id)
        path_parts = path.split('/')
        base_path = '/'.join(path_parts[:-2])  # Remove camera_id and group_id
       
        # Always load max_input_frames frames
        total_frames_needed = self.max_input_frames + self.target_frames
        if len(frame_indices) > total_frames_needed:
            rng = np.random.RandomState(idx)
            start_idx = rng.randint(0, len(frame_indices) - total_frames_needed + 1)
            frame_indices = frame_indices[start_idx:start_idx + total_frames_needed]
        elif len(frame_indices) < total_frames_needed:
            raise ValueError(f"Not enough frames available. Need {total_frames_needed} frames but only have {len(frame_indices)}")
        
        # source images
        c2ws, intrs, rgbs, bg_colors, masks, landmarks_list = [], [], [], [], [], []
        teeth_bs_pth = os.path.join(base_path, "tracked_teeth_bs.npz")
        use_teeth = False
        if os.path.exists(teeth_bs_pth) and use_teeth:
            teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
        else:
            teeth_bs_lst = None

        for i, frame_idx in enumerate(frame_indices):
            # Get complete frame info from transforms.json first
            frame_info = self.get_frame_info(base_path, frame_idx, camera_id)
            
            # Build frame directory path using the camera_id from path
            frame_dir = os.path.join(self.root_dir, base_path, camera_id, "processed_data", f"{frame_idx:05d}")
            
            # Load processed data
            rgb, mask, intrinsic, landmarks, bg_color = self.load_processed_data(frame_dir)
            
            if len(rgb.shape) == 3: 
                rgb = rgb.unsqueeze(0)  # Add batch dimension -> [1, 3, H, W]
            
            if len(mask.shape) == 3: 
                mask = mask.unsqueeze(0)  # Add batch dimension -> [1, 1, H, W]
            
            # Load camera pose using complete frame info
            c2w, _ = self._load_pose(frame_info)
            
            # Load FLAME parameters
            flame_path = os.path.join(self.root_dir, base_path, camera_id, "flame_param", f"{frame_idx:05d}.npz")
            flame_param = self.load_flame_params(flame_path, i, teeth_bs_lst[0] if teeth_bs_lst is not None else None)
            canonical_flame_path = os.path.join(self.root_dir, base_path, camera_id, "canonical_flame_param.npz")
            canonical_flame_param = self.load_flame_params(canonical_flame_path, 0, teeth_bs_lst[0] if teeth_bs_lst is not None else None)
            
            c2ws.append(c2w)
            intrs.append(intrinsic)
            rgbs.append(rgb)
            bg_colors.append(bg_color.item())
            masks.append(mask)
            landmarks_list.append(landmarks)  # Add landmarks to the list
            
        c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W] 
        bg_colors = torch.tensor(np.array(bg_colors), dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
        landmarks = torch.stack(landmarks_list, dim=0)  # [N, X, 2] 
        
        all_flame_params = defaultdict(list)
        for i, frame_idx in enumerate(frame_indices):
            flame_path = os.path.join(self.root_dir, base_path, camera_id, "flame_param", f"{frame_idx:05d}.npz")
            flame_param = self.load_flame_params(flame_path, i, teeth_bs_lst[0] if teeth_bs_lst is not None else None)
            canonical_flame_path = os.path.join(self.root_dir, base_path, camera_id, "canonical_flame_param.npz")
            canonical_flame_param = self.load_flame_params(canonical_flame_path, 0, teeth_bs_lst[0] if teeth_bs_lst is not None else None)
            
            for k, v in flame_param.items():
                all_flame_params[k].append(v)
        
        for k, v in all_flame_params.items():
            all_flame_params[k] = torch.stack(v)  # [N, ...]
        
        canonical_flame_param = self.load_flame_params(canonical_flame_path, 0, teeth_bs_lst[0] if teeth_bs_lst is not None else None)
        all_flame_params['betas'] = canonical_flame_param['shape'].expand(len(frame_indices), -1)  # [N, 300]

        ret = {
            'uid': uid,
            'c2ws': c2ws[:-self.target_frames],
            'target_c2ws': c2ws[-self.target_frames:],
            'intrs': intrs[:-self.target_frames],
            'target_intrs': intrs[-self.target_frames:],
            'rgbs': rgbs[:-self.target_frames],
            'target_rgbs': rgbs[-self.target_frames:],
            'bg_colors': bg_colors[:-self.target_frames],
            'target_bg_colors': bg_colors[-self.target_frames:],
            'masks': masks[:-self.target_frames],
            'target_masks': masks[-self.target_frames:],
            'landmarks': landmarks[:-self.target_frames],
            'target_landmarks': landmarks[-self.target_frames:],
            'max_input_frames': self.max_input_frames,
            'is_val': self.is_val,
        }
        
        for k, v in all_flame_params.items():
            ret[f'input_{k}'] = v[:-self.target_frames]  # [N_input, ...]
            ret[f'target_{k}'] = v[-self.target_frames:]  # [N_target, ...]
        
        assert ret['c2ws'].shape[0] == self.max_input_frames
        assert ret['target_c2ws'].shape[0] == self.target_frames
        assert ret['intrs'].shape[0] == self.max_input_frames
        assert ret['target_intrs'].shape[0] == self.target_frames
        assert ret['rgbs'].shape[0] == self.max_input_frames, f"{ret['rgbs'].shape}"
        assert ret['target_rgbs'].shape[0] == self.target_frames
        assert ret['bg_colors'].shape[0] == self.max_input_frames
        assert ret['target_bg_colors'].shape[0] == self.target_frames
        assert ret['masks'].shape[0] == self.max_input_frames
        assert ret['target_masks'].shape[0] == self.target_frames
        assert ret['landmarks'].shape[0] == self.max_input_frames
        assert ret['target_landmarks'].shape[0] == self.target_frames

        return ret


def collate_fn(batch):
    """Collate function for DataLoader to handle batch-level input frames.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Collated batch with consistent input frames
    """
    # Generate random input frames for this batch
    if not batch[0].get('is_val', False):  # Check if we're in validation mode
        # Extract sequence number from the first sample's uid path
        # e.g., from "sequence_EXP-2-eyes_part-5/304/cam_01/00001" extract "00001"
        sequence_path = batch[0]['uid'][0]
        sequence_num = int(sequence_path.split('/')[-1])  # Get the last number in the path
        
        # Use sequence number as seed for random number generation
        rng = np.random.RandomState(sequence_num)
        
        input_frames = rng.randint(2, batch[0]['max_input_frames'] + 1)
    else:
        input_frames = batch[0]['max_input_frames']

    # Process each sample in the batch
    processed_batch = []
    for sample in batch:
        # Get the actual frames we need for input parameters
        sample['rgbs'] = sample['rgbs'][:input_frames]
        sample['landmarks'] = sample['landmarks'][:input_frames]
        sample['c2ws'] = sample['c2ws'][:input_frames]
        sample['intrs'] = sample['intrs'][:input_frames]
        sample['bg_colors'] = sample['bg_colors'][:input_frames]
        sample['masks'] = sample['masks'][:input_frames]
        
        for key in list(sample.keys()):
            if key.startswith('input_'):
                if isinstance(sample[key], torch.Tensor) and sample[key].shape[0] > input_frames:
                    sample[key] = sample[key][:input_frames]
        
        processed_batch.append(sample)

    # Stack all tensors
    collated = {}
    for key in processed_batch[0].keys():
        if isinstance(processed_batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in processed_batch])
        else:
            collated[key] = [sample[key] for sample in processed_batch]
    
    return collated