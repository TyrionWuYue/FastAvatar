# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. All rights reserved.
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

import os
import traceback
import time
import torch
import argparse
import mcubes
import json
import numpy as np
from PIL import Image
from glob import glob
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from safetensors.torch import load_file
from accelerate.logging import get_logger
from collections import defaultdict
import cv2

from FastAvatar.runners.infer.head_utils import prepare_motion_seqs, preprocess_image, load_flame_params
from FastAvatar.runners.infer.base_inferrer import Inferrer
from FastAvatar.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from FastAvatar.runners import REGISTRY_RUNNERS
from FastAvatar.runners.infer.flame_tracking_multi_image import FlameTrackingMultiImage
from FastAvatar.runners.infer.flame_tracking_single_image import FlameTrackingSingleImage

logger = get_logger(__name__)

def parse_configs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)
    
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child)

        cfg.save_tmp_dump = os.path.join("infer_results", 'save_tmp', _relative_path)
        cfg.image_dump = os.path.join("infer_results", 'images', _relative_path)
        cfg.video_dump = os.path.join("infer_results", 'videos', _relative_path)
        
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("infer_results", cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault("image_dump", os.path.join("infer_results", cli_cfg.model_name, 'images'))
        cfg.setdefault('video_dump', os.path.join("infer_results", cli_cfg.model_name, 'videos'))
    
    cfg.merge_with(cli_cfg)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str

    [default]
    render_views: int
    render_fps: int
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')
    cfg.setdefault('max_single_frame_render', 8)  # Default max frames for single frame rendering
    cfg.setdefault('mode', 'Monocular')  # Default mode: Monocular or MultiView
    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video, "export_video should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register(name="infer.fastavatar")
class FastAvatarInferrer(Inferrer):

    EXP_TYPE = "fastavatar"

    def __init__(self):
        super().__init__()
        
        self.cfg = parse_configs()
        self.model: FastAvatarInferrer = self._build_model(self.cfg).to(self.device)
        
        # Initialize tracking based on mode
        self.mode = self.cfg.get('mode', 'Monocular')
        print(f"Initializing FastAvatar inference in {self.mode} mode")
        
        if self.mode == 'MultiView':
            # For MultiView mode, use single image tracking
            self.flametracking = FlameTrackingSingleImage(output_dir='infer_results/tracking_output',
                                                 alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
                                                 vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
                                                 human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
                                                 facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
                                                 detect_iris_landmarks=True,
                                                 args = self.cfg)
        else:
            # For Monocular mode, use multi-image tracking
            self.flametracking = FlameTrackingMultiImage(output_dir='infer_results/tracking_output',
                                                 alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
                                                 vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
                                                 human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
                                                 facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
                                                 detect_iris_landmarks=True,
                                                 args = self.cfg)

    def _build_model(self, cfg):
        from FastAvatar.models.modeling_FastAvatar import ModelFastAvatar
        
        model = ModelFastAvatar(**cfg.model)
        
        # Load base model weights
        resume = os.path.join(cfg.model_name, 'model.safetensors')
        print("==="*48)
        print(f"Loading base model from {resume}")
        if resume.endswith('.safetensors'):
            ckpt = load_file(resume, device='cpu')
        else:
            ckpt = torch.load(resume, map_location='cpu')
        
        # Load base model state dict
        model.load_state_dict(ckpt, strict=False)
        print("Finished loading base model weights from:", resume)
        
        print("==="*16*3)
        return model
    
    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)
    
    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def add_audio_to_video(self, video_path, out_path, audio_path):
        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
        print(f"Audio added successfully at {out_path}")
    
    def save_imgs_2_video(self, img_lst, v_pth, fps):
        from moviepy.editor import ImageSequenceClip
        images = [image.astype(np.uint8) for image in img_lst]
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(v_pth, codec='libx264')
        print(f"Video saved successfully at {v_pth}")

    def preprocess_video_input(self, video_path, inference_N_frames=8):
        """
        Preprocess video input by extracting frames and randomly sampling N frames
        
        Args:
            video_path: Path to input video file
            inference_N_frames: Number of frames to extract for inference
            
        Returns:
            str: Path to directory containing extracted frames
        """
        print(f"\nPreprocessing video input: {video_path}")
        print(f"Target number of frames: {inference_N_frames}")
        
        # Create temporary directory for extracted frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(self.cfg.save_tmp_dump, f'{video_name}_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Check if frames already exist
        existing_frames = glob(os.path.join(frames_dir, "*.png"))
        if len(existing_frames) >= inference_N_frames:
            print(f"Found {len(existing_frames)} existing frames, using them")
            return frames_dir
        
        # Extract all frames from video
        print("Extracting frames from video...")
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"Video properties: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
            
            # Extract all frames first
            all_frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Extracted {frame_count} frames...")
            
            cap.release()
            print(f"Successfully extracted {len(all_frames)} frames")
            
            # Randomly sample N frames
            if len(all_frames) <= inference_N_frames:
                print(f"Video has {len(all_frames)} frames, using all frames")
                selected_indices = list(range(len(all_frames)))
            else:
                # Use numpy random for reproducible sampling
                np.random.seed(42)  # Fixed seed for reproducibility
                selected_indices = np.random.choice(len(all_frames), inference_N_frames, replace=False)
                selected_indices = sorted(selected_indices)  # Sort for consistent ordering
                print(f"Randomly selected {inference_N_frames} frames from {len(all_frames)} total frames")
            
            # Save selected frames
            for i, frame_idx in enumerate(selected_indices):
                frame = all_frames[frame_idx]
                frame_path = os.path.join(frames_dir, f"{i:05d}.png")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            print(f"Saved {len(selected_indices)} frames to {frames_dir}")
            
            # Clean up memory
            del all_frames
            
            return frames_dir
            
        except Exception as e:
            print(f"Error extracting frames from video: {e}")
            raise
    
    def load_input_flame_params(self, tracking_output_dir, inference_N_frames=None):
        """Load FLAME parameters from multi-image tracking output"""
        flame_params = defaultdict(list)
        print(tracking_output_dir)
        # Load canonical FLAME parameters
        canonical_flame_path = os.path.join(tracking_output_dir, 'canonical_flame_param.npz')
        if os.path.exists(canonical_flame_path):
            canonical_params = np.load(canonical_flame_path)
            # Use canonical shape for all frames
            canonical_shape = torch.FloatTensor(canonical_params['shape'])
        else:
            print(f"Warning: Canonical FLAME parameters not found at {canonical_flame_path}")
            # Use default shape if canonical not available
            canonical_shape = torch.zeros(300)
        
        # Load per-frame FLAME parameters
        flame_param_dir = os.path.join(tracking_output_dir, 'flame_param')
        if not os.path.exists(flame_param_dir):
            raise FileNotFoundError(f"FLAME parameter directory not found at {flame_param_dir}. Tracking may have failed.")
        
        flame_param_files = sorted(glob(os.path.join(flame_param_dir, '*.npz')))
        if not flame_param_files:
            raise FileNotFoundError(f"No FLAME parameter files found in {flame_param_dir}")
        
        print(f"Found {len(flame_param_files)} FLAME parameter files")
        
        # If inference_N_frames is specified, only load the first N frames
        original_count = len(flame_param_files)
        if inference_N_frames is not None and len(flame_param_files) > inference_N_frames:
            flame_param_files = flame_param_files[:inference_N_frames]
            print(f"Using first {inference_N_frames} FLAME parameter files (out of {original_count} total)")
        
        for flame_file in flame_param_files:
            frame_params = load_flame_params(flame_file)
            for k, v in frame_params.items():
                if k != 'shape':  # shape is handled by canonical params
                    flame_params[k].append(v)
        
        # Stack all parameters to match training format: [N_frames, ...]
        for k in flame_params:
            flame_params[k] = torch.stack(flame_params[k])  # [N_frames, ...]
        
        # Use canonical shape for all frames: [N_frames, 300]
        num_frames = len(flame_param_files)
        flame_params['betas'] = canonical_shape.expand(num_frames, -1)  # [N_frames, 300]
        
        # Add batch dimension to match training format: [1, N_frames, ...]
        for k in flame_params:
            flame_params[k] = flame_params[k].unsqueeze(0)  # [1, N_frames, ...]
        
        print(f"Loaded FLAME parameters for {num_frames} frames")
        return flame_params
        
    def process_monocular_input(self, input_path, inference_N_frames):
        """Process input using multi-image FLAME tracking (Monocular mode)"""
        # Create temporary directory for tracking
        tracking_output_dir = os.path.join(self.cfg.save_tmp_dump, 'input_tracking')
        
        # Clean existing tracking output to avoid conflicts
        if os.path.exists(tracking_output_dir):
            import shutil
            shutil.rmtree(tracking_output_dir)
            print(f"Cleaned existing tracking output directory: {tracking_output_dir}")
        
        os.makedirs(tracking_output_dir, exist_ok=True)
        
        # Run multi-image FLAME tracking
        return_code = self.flametracking.preprocess(input_path, max_frames=inference_N_frames)
        assert (return_code == 0), "flametracking preprocess failed!"
        return_code = self.flametracking.optimize()
        assert (return_code == 0), "flametracking optimize failed!"
        return_code, output_dir = self.flametracking.export()
        assert (return_code == 0), "flametracking export failed!"
        
        return output_dir

    def process_multiview_input(self, input_path, inference_N_frames):
        """Process input using single image FLAME tracking for each view (MultiView mode)"""
        # Create temporary directory for tracking
        tracking_output_dir = os.path.join(self.cfg.save_tmp_dump, 'input_tracking_multiview')
        
        # Clean existing tracking output to avoid conflicts
        if os.path.exists(tracking_output_dir):
            import shutil
            shutil.rmtree(tracking_output_dir)
            print(f"Cleaned existing tracking output directory: {tracking_output_dir}")
        
        os.makedirs(tracking_output_dir, exist_ok=True)
        
        # Get all image files from input path
        if os.path.isfile(input_path):
            image_files = [input_path]
        else:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob(os.path.join(input_path, ext)))
                image_files.extend(glob(os.path.join(input_path, ext.upper())))
            image_files = sorted(image_files)
        
        if not image_files:
            raise ValueError(f"No image files found in {input_path}")
        
        # Limit number of frames if specified
        if inference_N_frames is not None and len(image_files) > inference_N_frames:
            image_files = image_files[:inference_N_frames]
            print(f"Limited to {inference_N_frames} frames (out of {len(image_files)} total)")
        
        print(f"Processing {len(image_files)} images in MultiView mode")
        
        # Process each image individually using single image tracking
        all_flame_params = []
        all_camera_params = []
        
        for i, image_file in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
            
            # Update output directory for this view
            view_output_dir = os.path.join(tracking_output_dir, f'view_{i:03d}')
            self.flametracking.output_dir = view_output_dir
            self.flametracking.output_preprocess = os.path.join(view_output_dir, 'preprocess')
            self.flametracking.output_tracking = os.path.join(view_output_dir, 'tracking')
            self.flametracking.output_export = os.path.join(view_output_dir, 'export')
            
            # Process single image
            return_code = self.flametracking.preprocess(image_file)
            if return_code != 0:
                print(f"Warning: Failed to preprocess image {image_file}, skipping...")
                continue
                
            return_code = self.flametracking.optimize()
            if return_code != 0:
                print(f"Warning: Failed to optimize image {image_file}, skipping...")
                continue
                
            return_code, export_dir = self.flametracking.export()
            if return_code != 0:
                print(f"Warning: Failed to export image {image_file}, skipping...")
                continue
            
            # Load FLAME parameters for this view
            view_flame_params = self.load_input_flame_params(export_dir, 1)  # Single frame
            all_flame_params.append(view_flame_params)
            
            # Load camera parameters for this view
            view_intrs, view_c2ws = self.load_input_camera_params(export_dir, 1)  # Single frame
            all_camera_params.append((view_intrs, view_c2ws))
        
        if not all_flame_params:
            raise RuntimeError("No images were successfully processed")
        
        # Combine all views into a single output directory
        combined_output_dir = os.path.join(tracking_output_dir, 'combined')
        os.makedirs(combined_output_dir, exist_ok=True)
        
        # Combine FLAME parameters from all views
        combined_flame_params = {}
        for key in all_flame_params[0].keys():
            combined_flame_params[key] = torch.cat([params[key] for params in all_flame_params], dim=1)
        
        # Save combined FLAME parameters
        flame_param_dir = os.path.join(combined_output_dir, 'flame_param')
        os.makedirs(flame_param_dir, exist_ok=True)
        
        # Save canonical shape (use first view's shape)
        canonical_shape = combined_flame_params['betas'][0, 0]  # [300]
        np.savez(os.path.join(combined_output_dir, 'canonical_flame_param.npz'), shape=canonical_shape.numpy())
        
        # Save per-frame FLAME parameters
        for i in range(combined_flame_params['betas'].shape[1]):
            frame_params = {k: v[0, i] for k, v in combined_flame_params.items()}
            np.savez(os.path.join(flame_param_dir, f'{i:05d}.npz'), **{k: v.numpy() for k, v in frame_params.items()})
        
        # Combine camera parameters
        combined_intrs = torch.cat([intrs for intrs, _ in all_camera_params], dim=1)  # [1, N_views, 4, 4]
        combined_c2ws = torch.cat([c2ws for _, c2ws in all_camera_params], dim=1)    # [1, N_views, 4, 4]
        
        # Create transforms.json for combined views
        transforms_data = {
            'camera_angle_x': 0.6911112070083618,
            'frames': []
        }
        
        for i in range(combined_intrs.shape[1]):
            intr = combined_intrs[0, i]
            c2w = combined_c2ws[0, i]
            
            # Convert back to OpenGL convention for transforms.json
            c2w_gl = c2w.clone()
            c2w_gl[:3, 1:3] *= -1
            
            frame_data = {
                'file_path': f'./images/{i:05d}.png',
                'fl_x': float(intr[0, 0]),
                'fl_y': float(intr[1, 1]),
                'cx': float(intr[0, 2]),
                'cy': float(intr[1, 2]),
                'transform_matrix': c2w_gl.tolist()
            }
            transforms_data['frames'].append(frame_data)
        
        # Save transforms.json
        with open(os.path.join(combined_output_dir, 'transforms.json'), 'w') as f:
            json.dump(transforms_data, f, indent=2)
        
        # Copy processed data from first view (they should be similar)
        first_view_export = os.path.join(tracking_output_dir, 'view_000', 'export')
        if os.path.exists(first_view_export):
            import shutil
            shutil.copytree(first_view_export, os.path.join(combined_output_dir, 'export'), dirs_exist_ok=True)
        
        print(f"Successfully combined {len(all_flame_params)} views into {combined_output_dir}")
        return combined_output_dir

    def load_input_camera_params(self, tracking_output_dir, N_input):
        """Load camera parameters for input frames from tracking output"""
        transforms_path = os.path.join(tracking_output_dir, 'transforms.json')
        
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"transforms.json not found at {transforms_path}, FLAME tracking incomplete")
        
        # Load transforms.json
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
        
        frames = transforms_data['frames']
        
        # Limit to N_input frames
        if len(frames) > N_input:
            frames = frames[:N_input]
        elif len(frames) < N_input:
            raise ValueError(f"transforms.json only contains {len(frames)} frames, but need {N_input}")
        
        input_intrs_list = []
        input_c2ws_list = []
        
        for frame in frames:
            # Load intrinsics
            intrinsic = torch.eye(4)
            intrinsic[0, 0] = frame["fl_x"]
            intrinsic[1, 1] = frame["fl_y"] 
            intrinsic[0, 2] = frame["cx"]
            intrinsic[1, 2] = frame["cy"]
            input_intrs_list.append(intrinsic)
            
            # Load c2w (camera-to-world transformation)
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w[:3, 1:3] *= -1  # Convert from OpenGL to OpenCV convention
            input_c2ws_list.append(c2w)
        
        # Stack into tensors with batch dimension
        input_intrs = torch.stack(input_intrs_list).unsqueeze(0)  # [1, N_input, 4, 4]
        input_c2ws = torch.stack(input_c2ws_list).unsqueeze(0)    # [1, N_input, 4, 4]
        
        print(f"Successfully loaded camera parameters for {len(frames)} input frames")
        return input_intrs, input_c2ws

    def infer(self):
        # Get inference_N_frames from config (default to 8)
        inference_N_frames = self.cfg.get('inference_N_frames', 8)
        
        # Check if input is a video file
        if os.path.isfile(self.cfg.image_input) and self.cfg.image_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video input - preprocess video first
            print(f"Detected video input: {self.cfg.image_input}")
            print(f"Using inference_N_frames: {inference_N_frames}")
            
            # Preprocess video to extract frames
            frames_dir = self.preprocess_video_input(self.cfg.image_input, inference_N_frames)
            input_path = frames_dir
            omit_prefix = frames_dir
        else:
            # Image or folder input
            if os.path.isfile(self.cfg.image_input):
                image_paths = [self.cfg.image_input]
                omit_prefix = os.path.dirname(self.cfg.image_input)
            else:
                image_paths = glob(os.path.join(self.cfg.image_input, "*.png"))
                omit_prefix = self.cfg.image_input
            input_path = self.cfg.image_input

        # Initialize lists to store processed data
        c2ws, rgbs, bg_colors, masks = [], [], [], []

        # Process input images based on mode
        if self.mode == 'MultiView':
            print(f"\nProcessing input with MultiView mode using single image FLAME tracking...")
            output_dir = self.process_multiview_input(input_path, inference_N_frames)
        else:
            print(f"\nProcessing input with Monocular mode using multi-image FLAME tracking...")
            output_dir = self.process_monocular_input(input_path, inference_N_frames)

        print(f"FLAME tracking completed. Output directory: {output_dir}")
        
        # Load input FLAME parameters
        input_flame_params = self.load_input_flame_params(output_dir, inference_N_frames)
        print(f"Input FLAME parameters loaded successfully for {inference_N_frames} frames")

        # Load processed data from tracking output (same as training)
        # processed_data is inside the output directory
        processed_data_dir = os.path.join(output_dir, 'processed_data')
        print(f"Looking for processed_data at: {processed_data_dir}")
        print(f"Directory exists: {os.path.exists(processed_data_dir)}")
        if os.path.exists(processed_data_dir):
            print(f"Directory contents: {os.listdir(processed_data_dir)}")
        
        if os.path.exists(processed_data_dir):
            print(f"Loading processed data from: {processed_data_dir}")
            
            # Get all frame directories
            frame_dirs = []
            for item in os.listdir(processed_data_dir):
                item_path = os.path.join(processed_data_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    frame_dirs.append((int(item), item_path))
            
            # Sort by frame number
            frame_dirs.sort(key=lambda x: x[0])
            
            if not frame_dirs:
                print(f"No frame directories found in {processed_data_dir}")
                return
            
            # Limit to inference_N_frames if specified
            original_count = len(frame_dirs)
            if inference_N_frames is not None and len(frame_dirs) > inference_N_frames:
                frame_dirs = frame_dirs[:inference_N_frames]
                print(f"Limited to {inference_N_frames} frames (out of {original_count} total)")
            else:
                print(f"Found {len(frame_dirs)} frames")
            
            for frame_num, frame_dir in frame_dirs:
                try:
                    # Load processed data files
                    rgb_path = os.path.join(frame_dir, 'rgb.npy')
                    mask_path = os.path.join(frame_dir, 'mask.npy')
                    intr_path = os.path.join(frame_dir, 'intrs.npy')
                    bg_color_path = os.path.join(frame_dir, 'bg_color.npy')
                    
                    # Check if all required files exist
                    required_files = [rgb_path, mask_path, intr_path, bg_color_path]
                    for file_path in required_files:
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Required file not found: {file_path}")
                    
                    # Load data
                    rgb = torch.from_numpy(np.load(rgb_path)).float()
                    mask = torch.from_numpy(np.load(mask_path)).float()
                    intr = torch.from_numpy(np.load(intr_path)).float()
                    bg_color = np.load(bg_color_path)  # 保持为 numpy array
                    
                    # Add batch dimension if needed
                    if len(rgb.shape) == 3:
                        rgb = rgb.unsqueeze(0)  # [1, 3, H, W]
                    if len(mask.shape) == 3:
                        mask = mask.unsqueeze(0)  # [1, 1, H, W]
                    
                    print(f"Frame {frame_num} loaded - RGB shape: {rgb.shape}, Mask shape: {mask.shape}")

                    # Collect data
                    rgbs.append(rgb)
                    masks.append(mask)
                    bg_colors.append(float(bg_color))  # 转换为 float
                    
                except Exception as e:
                    print(f"Error loading frame {frame_num}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    traceback.print_exc()
        else:
            print(f"Warning: processed_data directory not found at {processed_data_dir}")
            print("This suggests that process_data_augmentation was not run properly")
            return
        
        # Prepare motion sequences
        print(f"Preparing motion sequences from: {self.cfg.motion_seqs_dir}")
        motion_seqs = prepare_motion_seqs(
            motion_seqs_dir=self.cfg.motion_seqs_dir,
            image_folder=None,
            save_root=self.cfg.save_tmp_dump,
            fps=self.cfg.motion_video_read_fps,
            bg_color=1.0,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_image_res=self.cfg.render_size,
            need_mask=self.cfg.get("motion_img_need_mask", False),
            multiply=16,
            vis_motion=self.cfg.get("vis_motion", False),
            test_sample=self.cfg.get("test_sample", False),
            cross_id=self.cfg.get("cross_id", False)
        )
        print("Motion sequences prepared successfully")

        c2ws.append(motion_seqs["c2ws"])  # Already has batch dim

        # Use input FLAME parameters for shape, motion FLAME parameters for pose/expression
        inf_flame_params = motion_seqs["flame_params"]
        
        # Stack all collected data (maintaining batch dimension)
        target_c2ws = torch.cat(c2ws, dim=1)  # [1, N_target, 4, 4] - for rendering
        target_intrs = motion_seqs["intrs"]  # Already has batch dim [1, N_target, 4, 4] - for rendering
        
        # Stack input data from processed_data (following disorder_video_head.py pattern)
        if not rgbs:
            raise RuntimeError(f"No frames were successfully loaded. rgbs list is empty. processed_data_dir: {processed_data_dir}")
        
        print(f"Successfully loaded {len(rgbs)} frames")
        rgbs = torch.cat(rgbs, dim=0)  # [N_input, 3, H, W]
        rgbs = rgbs.unsqueeze(0)  # [1, N_input, 3, H, W]
        masks = torch.cat(masks, dim=0)  # [N_input, 1, H, W]
        masks = masks.unsqueeze(0)  # [1, N_input, 1, H, W]
        
        # Load input intrinsics from processed_data
        input_intrs_list = []
        for frame_num, frame_dir in frame_dirs:
            intr_path = os.path.join(frame_dir, 'intrs.npy')
            intr = torch.from_numpy(np.load(intr_path)).float()
            input_intrs_list.append(intr)
        input_intrs = torch.stack(input_intrs_list, dim=0)  # [N_input, 4, 4]
        input_intrs = input_intrs.unsqueeze(0)  # [1, N_input, 4, 4]

        # Load camera poses from tracking output
        _, input_c2ws = self.load_input_camera_params(output_dir, len(frame_dirs))

        # input bg_colors: only use processed_data's float values
        input_bg_colors = torch.tensor(np.array(bg_colors), dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N_input, 3]
        input_bg_colors = input_bg_colors.unsqueeze(0)  # [1, N_input, 3]

        # target bg_colors: use motion_seqs["bg_colors"] directly
        target_bg_colors = motion_seqs["bg_colors"]  # [1, N_target, 3]

        # Stack FLAME parameters (maintaining batch dimension)
        for k in inf_flame_params:
            inf_flame_params[k] = torch.cat([inf_flame_params[k]], dim=1)  # [1, N_target, ...]
        
        n_inf = target_c2ws.shape[1]

        # Move all tensors to GPU
        device = self.device
        rgbs = rgbs.to(device)
        target_c2ws = target_c2ws.to(device)
        target_intrs = target_intrs.to(device)
        target_bg_colors = target_bg_colors.to(device)
        input_c2ws = input_c2ws.to(device)
        input_intrs = input_intrs.to(device)
        masks = masks.to(device)
        inf_flame_params = {k: v.to(device) for k, v in inf_flame_params.items()}
        input_flame_params = {k: v.to(device) for k, v in input_flame_params.items()}

        # Assert that input FLAME parameters betas frame count matches input frame count
        input_frame_count = rgbs.shape[1]  # Number of input frames
        input_betas_frame_count = input_flame_params["betas"].shape[1]  # Number of frames in input betas
        assert input_frame_count == input_betas_frame_count, \
            f"Input frame count ({input_frame_count}) does not match input FLAME betas frame count ({input_betas_frame_count})"

        # Run model inference
        print("\nStarting model inference.........................")
        start_time = time.time()

        with torch.no_grad():
            frame_results = []
            
            # Get max_single_frame_render from config, default to 8
            max_single_frame_render = self.cfg.get("max_single_frame_render", 8)
            input_frame_count = rgbs.shape[1]
            
            print(f"Input frame count: {input_frame_count}")
            print(f"Max single frame render: {max_single_frame_render}")
                
            # Multi-frame inference
            multi_frame_start_time = time.time()
            res = self.model.infer_images(
                rgbs,
                input_c2ws,
                input_intrs,
                target_c2ws,
                target_intrs,
                target_bg_colors,
                input_flame_params=input_flame_params,
                inf_flame_params=inf_flame_params
            )
            multi_frame_total_time = time.time() - multi_frame_start_time
            print(f"Multi-frame rendering completed:")
            print(f"  - Total time: {multi_frame_total_time:.2f} seconds")
            
            frame_results.append(res["comp_rgb"].detach().cpu().numpy())
                
            # Combine results horizontally
            combined_results = []
            for i in range(len(frame_results[0])):
                frame_row = []
                for result in frame_results:
                    frame_row.append(result[i])
                combined_frame = np.hstack(frame_row)
                combined_results.append(combined_frame)
            rgb = np.array(combined_results)
        
        total_inference_time = time.time() - start_time
        print(f"Total inference completed in {total_inference_time:.2f} seconds")

        # Process results
        print("\nProcessing inference results...")
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        only_pred = rgb

        # Save results
        print("\nSaving results...")
        if os.path.isfile(self.cfg.image_input) and self.cfg.image_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # For video input, use the video name
            uid = os.path.splitext(os.path.basename(self.cfg.image_input))[0]
        elif os.path.isfile(self.cfg.image_input):
            # For single image input, use the original logic
            uid = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        else:
            # For folder input, use the folder name
            uid = os.path.basename(self.cfg.image_input)
        
        dump_video_path = os.path.join(self.cfg.video_dump, f'{uid}.mp4')
        dump_image_dir = os.path.join(self.cfg.image_dump, f'{uid}')
        dump_tmp_dir = os.path.join(self.cfg.image_dump, "tmp_res")
        
        os.makedirs(dump_image_dir, exist_ok=True)
        os.makedirs(dump_tmp_dir, exist_ok=True)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        self.save_imgs_2_video(rgb, dump_video_path, self.cfg.render_fps)
        print(f"Video saved to: {dump_video_path}")

        # Add audio if available
        base_vid = self.cfg.motion_seqs_dir.strip('/').split('/')[-1]
        audio_path = os.path.join(self.cfg.motion_seqs_dir, base_vid + ".wav")
        if os.path.exists(audio_path):
            dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
            self.add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
            print(f"Audio added to video: {dump_video_path_wa}")

        # Save individual frames (disabled by default)
        if False and dump_image_dir is not None:
            print("\nSaving individual frames...")
            for i in range(rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(only_pred[i]).save(save_file)

        print("\nInference and saving completed successfully!")