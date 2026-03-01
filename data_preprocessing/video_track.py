import glob
import os
import cv2
import time
import tyro
import torch
import torchvision
import torch.nn.functional as F
import shutil
import json
import argparse
import subprocess

import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import yaml
from PIL import Image
from external.vgghead_detector import VGGHeadDetector
from external.human_matting import StyleMatteEngine as HumanMattingEngine
from insightface.app import FaceAnalysis
from external.landmark_detection.FaceBoxesV2.faceboxes_detector import FaceBoxesDetector
from external.landmark_detection.infer_folder import Alignment
from vhap.config.base import BaseTrackingConfig
from vhap.export_as_nerf_dataset import (NeRFDatasetWriter,
                                         TrackedFLAMEDatasetWriter, split_json)

from vhap.model.tracker import GlobalTracker

# Define error codes for various processing failures.
ERROR_CODE = {'FailedToDetect': 1, 'FailedToOptimize': 2, 'FailedToExport': 3}

# Seed for reproducibility
SEED = 42


def expand_bbox(bbox, scale=1.1):
    """Expands the bounding box by a given scale."""
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    hight = ymax - ymin
    width = xmax - xmin
    extension_size = torch.sqrt(hight * width) * scale
    x_min_expanded = center_x - extension_size / 2
    x_max_expanded = center_x + extension_size / 2
    y_min_expanded = center_y - extension_size / 2
    y_max_expanded = center_y + extension_size / 2
    return torch.stack(
        [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded],
        dim=-1)

def load_config(src_folder: Path):
    """Load configuration from the given source folder."""
    config_file_path = src_folder / 'config.yml'
    if not config_file_path.exists():
        src_folder = sorted(
            src_folder.iterdir())[-1]  # Get the last modified folder
        config_file_path = src_folder / 'config.yml'
    assert config_file_path.exists(), f'File not found: {config_file_path}'

    config_data = yaml.load(config_file_path.read_text(), Loader=yaml.Loader)
    return src_folder, config_data

def run_landmark_detection_on_folder(input_folder, output_dir, device_id):
    """Run landmark detection on the entire folder using infer_folder.py approach."""
    try:
        # Create temporary directory for landmark detection
        temp_landmark_dir = os.path.join(output_dir, 'temp_landmark')
        os.makedirs(temp_landmark_dir, exist_ok=True)
        
        # Copy all images to temp directory for landmark detection
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        if not image_files:
            logger.error(f'No image files found in {input_folder}')
            return False
        
        # Sort image files
        image_files = sorted(image_files)
        
        # Copy images to temp directory with sequential naming
        for idx, img_path in enumerate(image_files):
            new_name = f'{idx:05d}.png'
            new_path = os.path.join(temp_landmark_dir, new_name)
            shutil.copy2(img_path, new_path)
        
        # Run landmark detection using infer_folder.py
        current_dir = os.getcwd()
        landmark_script_path = os.path.join(current_dir, 'external', 'landmark_detection', 'infer_folder.py')
        
        if not os.path.exists(landmark_script_path):
            logger.error(f'Landmark detection script not found: {landmark_script_path}')
            return False
        
        # Run landmark detection without changing directory
        cmd = [
            'python', landmark_script_path,
            '--folder_path', temp_landmark_dir
        ]
        
        logger.info(f'Running landmark detection: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode != 0:
            logger.error(f'Landmark detection failed: {result.stderr}')
            return False
        
        # Copy results back to output directory
        landmark_output_dir = os.path.join(output_dir, 'landmark')
        os.makedirs(landmark_output_dir, exist_ok=True)
        
        # Copy keypoint.json if it exists (now saved inside temp_landmark_dir)
        keypoint_json = os.path.join(temp_landmark_dir, 'keypoint.json')
        if os.path.exists(keypoint_json):
            shutil.copy2(keypoint_json, os.path.join(output_dir, 'keypoint.json'))
        
        # Clean up temp directory
        shutil.rmtree(temp_landmark_dir)
        
        logger.info(f'Successfully completed landmark detection for {len(image_files)} images')
        return True
        
    except Exception as e:
        logger.error(f'Error in landmark detection: {str(e)}')
        return False

def save_processed_data(save_dir, landmarks, intr, rgb, mask, bg_color):
    """Save all processed data to a single directory.
    
    Args:
        save_dir: Directory to save the data
        landmarks: Face landmarks
        intr: Camera intrinsic matrix
        rgb: Processed RGB image
        mask: Processed mask
        bg_color: Background color used for processing
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save landmarks
    np.savez(os.path.join(save_dir, 'landmark2d.npz'),
             face_landmark_2d=landmarks[None, ...])
    
    # Save camera intrinsics
    np.save(os.path.join(save_dir, 'intrs.npy'), intr.numpy())
    
    # Save processed image and mask
    # Ensure RGB values are in [0, 1] range before saving
    rgb_clamped = torch.clamp(rgb, 0.0, 1.0)
    np.save(os.path.join(save_dir, 'rgb.npy'), rgb_clamped.numpy())
    np.save(os.path.join(save_dir, 'mask.npy'), mask.numpy())
    np.save(os.path.join(save_dir, 'bg_color.npy'), bg_color)

def get_training_data(frame_dir):
    try:
        np.random.seed(SEED)

        # Load the processed images
        images_dir = os.path.join(frame_dir, 'images')
        if not os.path.exists(images_dir):
            logger.warning(f'Images directory not found: {images_dir}')
            return False
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        if not image_files:
            logger.warning(f'No image files found in {images_dir}')
            return False
        
        # Landmark2d data
        landmark_path = os.path.join(frame_dir, 'landmark2d', 'landmarks.npz')
        if not os.path.exists(landmark_path):
            logger.warning(f'Landmarks file not found: {landmark_path}')
            return False

        # transforms.json data
        transforms_path = os.path.join(os.path.dirname(frame_dir), 'transforms.json')
        if not os.path.exists(transforms_path):
            logger.warning(f'Transforms.json not found: {transforms_path}')
            return False
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
        
        # Load all landmarks for all frames
        landmark_data = np.load(landmark_path)
        all_landmarks = landmark_data['face_landmark_2d'][..., :2]  # (num_frames, 68, 2)
        
        # Verify that images and landmarks count match (strict check)
        num_landmarks = len(all_landmarks)
        num_images = len(image_files)
        if num_images != num_landmarks:
            logger.error(f'CRITICAL: Mismatch between images ({num_images}) and landmarks ({num_landmarks}). '
                         f'This should not happen if preprocessing was done correctly.')
            raise ValueError(f'Image-landmark count mismatch: {num_images} images vs {num_landmarks} landmarks')
        
        # Process each image using the corresponding landmarks
        for frame_idx, img_path in enumerate(image_files):
            try:
                logger.info(f'Processing frame {frame_idx + 1}/{len(image_files)}')
                
                # Get frame data from transforms
                if frame_idx < len(transforms_data['frames']):
                    frame_data = transforms_data['frames'][frame_idx]
                else:
                    # Use first frame data if not enough frames in transforms
                    frame_data = transforms_data['frames'][0]
                
                intr = torch.eye(4)
                intr[0, 0] = frame_data["fl_x"]
                intr[1, 1] = frame_data["fl_y"]
                intr[0, 2] = frame_data["cx"]
                intr[1, 2] = frame_data["cy"]
                intr = intr.float()
                
                # 1. Load image
                rgb = np.array(Image.open(img_path))
                rgb = rgb / 255.0
                
                # 2. Load mask
                img_name = os.path.basename(img_path)
                mask_path = os.path.join(frame_dir, 'mask', img_name)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path))
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]
                    if len(mask.shape) > 2:
                        mask = mask[:, :, 0]
                    mask = (mask > 0.5).astype(np.float32)
                else:
                    # If mask doesn't exist, create a default mask (all ones)
                    logger.warning(f'Mask file not found: {mask_path}, using default mask')
                    mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                
                # 3. Apply background color
                bg_color = random.choice([0.0, 0.5, 1.0])
                rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
                
                # 4. Use landmarks directly as [0, 1] normalized (no modifications)
                landmarks_normalized = all_landmarks[frame_idx][:, :2]  # (68, 2) already in [0, 1]
                
                # Convert to tensor format
                rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1)
                
                rgb = torch.clamp(rgb, 0.0, 1.0)
                
                # Create processed_data directory
                parent_dir = os.path.dirname(frame_dir)
                processed_data_dir = os.path.join(parent_dir, 'processed_data', f'{frame_idx:05d}')
                os.makedirs(processed_data_dir, exist_ok=True)
                
                save_processed_data(processed_data_dir, landmarks_normalized, intr, rgb, mask, bg_color)

            except Exception as e:
                logger.error(f'Error processing frame {frame_idx}: {str(e)}')
                continue
         
        # Delete redundant intermediate files
        preprocess_path = os.path.join(parent_dir, 'preprocess')
        if os.path.exists(preprocess_path):
            shutil.rmtree(preprocess_path)
        
        images_path = os.path.join(parent_dir, 'images')
        if os.path.exists(images_path):
            shutil.rmtree(images_path)

        fg_mask_path = os.path.join(parent_dir, 'fg_masks')
        if os.path.exists(fg_mask_path):
            shutil.rmtree(fg_mask_path)
        
        # Delete all JSON files except transforms.json
        json_files = glob.glob(os.path.join(parent_dir, '*.json'))
        for json_file in json_files:
            if os.path.basename(json_file) != 'transforms.json':
                os.remove(json_file)
                logger.info(f'Deleted JSON file: {json_file}')
        
        logger.info(f'Successfully applied data preprocessing to {frame_dir}')
        return True
    except Exception as e:
        logger.error(f'Error in data preprocessing for {frame_dir}: {str(e)}')
        return False
    finally:
        # Clean up memory after processing
        torch.cuda.empty_cache()
        import gc
        gc.collect()

class TrackingVideoImage:
    def __init__(
            self,
            output_dir: str,
            device_id: int,
            human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
            alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
            facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth'):
        
        logger.info(f"Output directory: {output_dir}")

        start_time = time.time()
        logger.info("Loading Pre-trained Models")

        self.output_dir = output_dir
        self.output_preprocess = os.path.join(output_dir, 'preprocess')
        os.makedirs(self.output_preprocess, exist_ok=True)
        self.output_tracking = os.path.join(output_dir, 'tracking')
        self.output_export = output_dir
        self.device_id = device_id
        self.device = f"cuda:{device_id}"

        # Set CUDA device and clear cache
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        self.face_app = FaceAnalysis()
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Load human matting model
        assert os.path.exists(
            human_matting_path), f'{human_matting_path} does not exist!'
        self.matting_engine = HumanMattingEngine(
            device=self.device, human_matting_path=human_matting_path)
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd'
        self.vgghead_encoder = VGGHeadDetector(
            device=self.device, vggheadmodel_path=vgghead_model_path)
        
        # Load alignment model for 68-point landmark detection
        assert os.path.exists(
            alignment_model_path), f'{alignment_model_path} does not exist!'
        args = argparse.Namespace()
        args.config_name = "alignment"
        args.model_path = alignment_model_path
        self.alignment = Alignment(args,
                                   alignment_model_path,
                                   dl_framework='pytorch',
                                   device_ids=[self.device_id])

        # Load face box detector model
        assert os.path.exists(
            facebox_model_path), f'{facebox_model_path} does not exist!'
        self.detector = FaceBoxesDetector('FaceBoxes', facebox_model_path,
                                          True, torch.device(self.device))
        
        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(f'Finished Loading Pre-trained Models. Time: '
                    f'{end_time - start_time:.2f}s')
     


    
    def video_preprocess(self, video_path, current_dir, image_size=512, target_fps=None, head_detect_freq=1):
        # Clean up existing directories to avoid stale files
        if os.path.exists(current_dir):
            shutil.rmtree(current_dir)
            logger.info(f"Cleaned up existing preprocess directory: {current_dir}")

        original_image_dir          = os.path.join(current_dir, 'original_images')
        output_image_dir            = os.path.join(current_dir, 'images')
        output_mask_dir             = os.path.join(current_dir, 'mask')
        output_alpha_map_dir        = os.path.join(current_dir, 'alpha_maps')
        cropped_landmark_output_dir = os.path.join(current_dir, 'landmark')
        landmark_output_dir         = os.path.join(current_dir, 'landmark2d')
        paste_back_output_dir       = os.path.join(current_dir, 'paste_back')
        tracker_output_dir          = os.path.join(current_dir, 'tracker_output')
        os.makedirs(original_image_dir, exist_ok=True)
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_alpha_map_dir, exist_ok=True)
        os.makedirs(cropped_landmark_output_dir, exist_ok=True)
        os.makedirs(landmark_output_dir, exist_ok=True)
        os.makedirs(paste_back_output_dir, exist_ok=True)
        os.makedirs(tracker_output_dir, exist_ok=True)
        cap = None
        if os.path.isdir(video_path):
            logger.info(f"Processing image directory: {video_path}")
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(video_path, ext)))
                image_files.extend(glob.glob(os.path.join(video_path, ext.upper())))
            
            image_files = sorted(image_files)
            if not image_files:
                logger.error(f"No images found in {video_path}")
                return 0
            
            total_frames = len(image_files)
            # For image sequences, we process all frames as requested
            original_fps = 30.0 
            is_video_file = False
        else:
            cap         = cv2.VideoCapture(video_path)
            total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            is_video_file = True
        
        # Calculate frame skip interval if target_fps is specified
        if target_fps is not None and target_fps > 0:
            frame_skip = max(1, int(original_fps / target_fps))
            logger.info(f"Original FPS: {original_fps:.2f}, Target FPS: {target_fps:.2f}, Frame skip: {frame_skip}")
        else:
            frame_skip = 1
            logger.info(f"Processing all frames (no frame skipping)")
        
        # First pass: detect all frames and find the maximum bounding box
        logger.info(f"First pass: Detecting faces with frequency {head_detect_freq} to find unified bbox...")
        all_bboxes = []
        frame_data = []
        
        # Get all frame indices after frame skipping
        frame_indices = list(range(0, total_frames, frame_skip))
        
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc="Detecting")):
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = frame[:,:,[2,1,0]]  # BGR to RGB
            else:
                # Read from image list
                if frame_idx >= len(image_files):
                    continue
                frame_path = image_files[frame_idx]
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                frame = frame[:,:,[2,1,0]]  # BGR to RGB

            frame_tensor = torch.from_numpy(frame).permute(2,0,1)  # HWC to CHW
            
            # Only detect every head_detect_freq frames
            if idx % head_detect_freq == 0:
                _, frame_bbox, _ = self.vgghead_encoder(frame_tensor, frame_idx)
                if frame_bbox is None:
                    logger.warning(f'No face detected in frame {frame_idx} of video {video_path}. Skipping this frame.')
                    continue
                all_bboxes.append(frame_bbox.cpu())
            
            # Always save frame data for second pass processing
            frame_data.append((frame_idx, frame_tensor))
        
        if len(all_bboxes) == 0:
            logger.error("No face detected in any frame!")
            return 0
        
        # Find the maximum bounding box that contains all detected faces
        all_bboxes_tensor = torch.stack(all_bboxes)  # (N, 4)
        unified_bbox_raw = torch.tensor([
            all_bboxes_tensor[:, 0].min().item(),  # min x_min
            all_bboxes_tensor[:, 1].min().item(),  # min y_min
            all_bboxes_tensor[:, 2].max().item(),  # max x_max
            all_bboxes_tensor[:, 3].max().item(),  # max y_max
        ]).long()
        
        # Expand the unified bounding box
        unified_bbox = expand_bbox(unified_bbox_raw, scale=1.65).long()
        logger.info(f"Using unified bbox: {unified_bbox.tolist()}")
        
        # Second pass: process all frames with the unified bounding box
        logger.info("Second pass: Processing frames with unified bbox...")
        exp_height = unified_bbox[3] - unified_bbox[1]
        exp_width = unified_bbox[2] - unified_bbox[0]
        
        # Use consecutive index starting from 0 for file naming
        processed_frames = []
        for consecutive_idx, (frame_idx, frame) in enumerate(tqdm(frame_data, desc="Processing")):
            cropped_frame = torchvision.transforms.functional.crop(
                    frame,
                    top     = unified_bbox[1],
                    left    = unified_bbox[0],
                    height  = exp_height,
                    width   = exp_width)
            cropped_frame   = torchvision.transforms.functional.resize(
                cropped_frame, (image_size, image_size), antialias=True)
            ori_image       = np.round(cropped_frame.cpu().permute(
                1, 2, 0).numpy()).astype(np.uint8)[:, :, (2, 1, 0)]
            # Use consecutive index for file naming (00000.png, 00001.png, ...)
            frame_name      = f'{consecutive_idx:05d}.png'
            cv2.imwrite(os.path.join(original_image_dir, frame_name), ori_image)
            cropped_frame, mask = self.matting_engine(cropped_frame / 255.0,
                                                          return_type='matting',
                                                          background_rgb=1.0)
            cropped_frame   = cropped_frame * 255.0
            saved_image     = np.round(cropped_frame.cpu().permute(
                1, 2, 0).numpy()).astype(np.uint8)[:, :, (2, 1, 0)]

            cv2.imwrite(os.path.join(output_image_dir, frame_name), saved_image)
            cv2.imwrite(os.path.join(output_mask_dir, frame_name),
                        np.array((mask.cpu() * 255.0)).astype(np.uint8))
            cv2.imwrite(
                os.path.join(output_alpha_map_dir, frame_name.replace('.png', '.jpg')),
                (np.ones_like(saved_image) * 255).astype(np.uint8))
            np.savez(os.path.join(paste_back_output_dir,
                                  f'{consecutive_idx:05d}.npz'), 
                                  **{
                                        'crop_anchor': np.array(unified_bbox.cpu())
                                  }
                                  )
            processed_frames.append((consecutive_idx, frame_name))
        
        # Step 3: Run batch landmark detection on all processed images
        logger.info('Step 3: Running batch landmark detection...')
        if not run_landmark_detection_on_folder(output_image_dir, current_dir, self.device_id):
            logger.error('Failed to run batch landmark detection')
            return
        
        # Step 4: Convert landmark results to the expected format
        logger.info('Step 4: Converting landmark results...')
        keypoint_json_path = os.path.join(current_dir, 'keypoint.json')
        if os.path.exists(keypoint_json_path):
            with open(keypoint_json_path, 'r') as f:
                keypoint_data = json.load(f)
            
            # For video processing, we need to create landmark data for all frames
            if processed_frames and keypoint_data:
                # First pass: collect valid frames (with landmarks) and remove invalid ones
                valid_frames_data = []  # List of (old_frame_name, landmarks, old_consecutive_idx)
                
                for consecutive_idx, frame_name in processed_frames:
                    if frame_name in keypoint_data:
                        landmarks = keypoint_data[frame_name]
                        valid_frames_data.append((frame_name, landmarks, consecutive_idx))
                    else:
                        logger.warning(f'No landmark data found for frame {frame_name}, removing this frame from dataset')
                        # Remove image, mask, and other files for frames without landmarks
                        files_to_remove = [
                            os.path.join(output_image_dir, frame_name),
                            os.path.join(output_mask_dir, frame_name),
                            os.path.join(original_image_dir, frame_name),
                            os.path.join(output_alpha_map_dir, frame_name.replace('.png', '.jpg')),
                            os.path.join(paste_back_output_dir, f'{consecutive_idx:05d}.npz'),
                        ]
                        for file_path in files_to_remove:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                logger.debug(f'Removed {file_path}')
                
                # Check if we have any valid landmarks
                if not valid_frames_data:
                    logger.error('No valid landmark data found for any frame. All frames failed detection.')
                    return
                
                logger.info(f'Successfully processed {len(valid_frames_data)}/{len(processed_frames)} frames (skipped {len(processed_frames) - len(valid_frames_data)} failed detections)')
                
                # Second pass: renumber valid frames to ensure consecutive numbering (0, 1, 2, ...)
                # and collect landmarks
                all_landmarks = []
                all_bounding_boxes = []
                
                for new_idx, (old_frame_name, landmarks, old_consecutive_idx) in enumerate(valid_frames_data):
                    new_frame_name = f'{new_idx:05d}.png'
                    
                    # Rename files if needed
                    if old_frame_name != new_frame_name:
                        # Rename image
                        old_img_path = os.path.join(output_image_dir, old_frame_name)
                        new_img_path = os.path.join(output_image_dir, new_frame_name)
                        if os.path.exists(old_img_path):
                            os.rename(old_img_path, new_img_path)
                        
                        # Rename mask
                        old_mask_path = os.path.join(output_mask_dir, old_frame_name)
                        new_mask_path = os.path.join(output_mask_dir, new_frame_name)
                        if os.path.exists(old_mask_path):
                            os.rename(old_mask_path, new_mask_path)
                        
                        # Rename original image
                        old_orig_path = os.path.join(original_image_dir, old_frame_name)
                        new_orig_path = os.path.join(original_image_dir, new_frame_name)
                        if os.path.exists(old_orig_path):
                            os.rename(old_orig_path, new_orig_path)
                        
                        # Rename alpha map
                        old_alpha_path = os.path.join(output_alpha_map_dir, old_frame_name.replace('.png', '.jpg'))
                        new_alpha_path = os.path.join(output_alpha_map_dir, new_frame_name.replace('.png', '.jpg'))
                        if os.path.exists(old_alpha_path):
                            os.rename(old_alpha_path, new_alpha_path)
                        
                        # Rename paste_back npz
                        old_paste_path = os.path.join(paste_back_output_dir, f'{old_consecutive_idx:05d}.npz')
                        new_paste_path = os.path.join(paste_back_output_dir, f'{new_idx:05d}.npz')
                        if os.path.exists(old_paste_path):
                            os.rename(old_paste_path, new_paste_path)
                    
                    # Normalize landmarks for landmark2d format
                    normalized_landmarks = np.zeros((len(landmarks), 3))
                    normalized_landmarks[:, :2] = np.array(landmarks) / image_size
                    
                    all_landmarks.append(normalized_landmarks)
                    all_bounding_boxes.append([])  # Empty bounding box for compatibility
                    
                    # Draw landmarks for visualization
                    image_draw = cv2.imread(os.path.join(output_image_dir, new_frame_name))
                    if image_draw is not None:
                        for num in range(len(landmarks)):
                            cv2.circle(image_draw, (round(landmarks[num][0]), round(landmarks[num][1])), 
                                    2, (0, 255, 0), -1)
                        cv2.imwrite(os.path.join(cropped_landmark_output_dir, new_frame_name), image_draw)
                
                if all_landmarks:
                    # Stack all landmarks into a single array
                    stacked_landmarks = np.stack(all_landmarks, axis=0)  # (num_frames, 68, 3)
                    stacked_bboxes = np.array(all_bounding_boxes)  # (num_frames, 0)
                    
                    # Create visibility array for all frames (all landmarks visible)
                    stacked_visibility = np.ones((len(all_landmarks), 68), dtype=bool)
                    
                    landmark_data = {
                        'bounding_box': stacked_bboxes,
                        'face_landmark_2d': stacked_landmarks,
                        'visibility': stacked_visibility,
                    }

                    # Save with the same naming as tracking.py
                    landmark_path = os.path.join(landmark_output_dir, 'landmarks.npz')
                    np.savez(landmark_path, **landmark_data)
                    logger.info(f'Successfully saved landmark2d data for {len(all_landmarks)} frames to {landmark_path}')

                    # Save 68-point landmark data (using first frame as main landmark)
                    first_frame_landmarks = all_landmarks[0][:, :2]  # Remove the third dimension
                    cropped_landmark_data_68 = {
                        'face_landmark_2d': first_frame_landmarks[None, ...],
                        'visibility': np.ones((1, 68), dtype=bool),
                    }
                    
                    cropped_landmark_path_68 = os.path.join(cropped_landmark_output_dir, 'landmark68.npz')
                    np.savez(cropped_landmark_path_68, **cropped_landmark_data_68)
                    logger.info(f'Successfully saved detected 68-point landmarks to {cropped_landmark_path_68}')
                    
                    torch.cuda.empty_cache()
                    return len(all_landmarks)  # Return number of valid frames
                else:
                    logger.error('No landmark data available for any frame')
                    torch.cuda.empty_cache()
                    return 0
            else:
                logger.warning(f'No processed images or landmark data available for landmark conversion')
                torch.cuda.empty_cache()
                return 0
        else:
            logger.warning('keypoint.json not found after landmark detection')
            torch.cuda.empty_cache()
            return 0
        
        torch.cuda.empty_cache()
        return 0

    def tracking_video(self, video_path, image_size=512, target_fps=None, head_detect_freq=1, min_valid_frames=0):
        current_dir = self.output_preprocess
        valid_frames_count = self.video_preprocess(video_path, current_dir, image_size, target_fps=target_fps, head_detect_freq=head_detect_freq)
        
        # Check if we have enough frames
        if valid_frames_count < min_valid_frames:
            logger.warning(f"Insufficient valid frames: {valid_frames_count} < {min_valid_frames}. Deleting output directory.")   
            cam_dir = os.path.dirname(self.output_dir)
            if os.path.exists(cam_dir) and len(str(cam_dir)) > 10: 
                 shutil.rmtree(cam_dir)
                 logger.info(f"Deleted camera directory: {cam_dir}")
            elif os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
                logger.info(f"Deleted output directory: {self.output_dir}")
            return

        tracker_output_dir = os.path.join(current_dir, 'tracker_output')
        self.output_tracking = tracker_output_dir  # Update the tracking directory path
        self._optimize(current_dir, tracker_output_dir)
        self._export()
        get_training_data(current_dir)

    def _optimize(self, preprocess_dir, output_tracking):
        """Optimize the tracking model using configuration data."""
        start_time = time.time()
        logger.info('Starting Optimization...')

        # try:
        tyro.extras.set_accent_color('bright_yellow')
        from yaml import safe_load
        with open("configs/vhap_tracking/base_tracking_config.yaml", 'r') as yml_f:
            config_data = safe_load(yml_f)
        config_data = tyro.from_yaml(BaseTrackingConfig, config_data)

        # Use the preprocess directory as the sequence directory
        # This is where the images, mask, landmark, etc. are located
        config_data.data.sequence = os.path.basename(preprocess_dir)
        # Use the directory containing preprocess as root folder
        config_data.data.root_folder = Path(os.path.dirname(preprocess_dir))

        if not os.path.exists(preprocess_dir):
            logger.error(f'Failed to load {preprocess_dir}')
            return ERROR_CODE['FailedToOptimize']

        config_data.exp.output_folder = Path(output_tracking)
        # Save config for later use in export
        self.tracking_config = config_data
        tracker = GlobalTracker(config_data)
        self.tracker_out_dir = tracker.out_dir
        tracker.optimize()

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(
            f'Finished Optimization. Time: {end_time - start_time:.2f}s')
        return 0
    
    def _export(self):
        """Export the tracking results to configured folder."""
        logger.info(f'Beginning export from {self.output_tracking}')
        
        try:
            if not os.path.exists(self.output_tracking):
                logger.error(f'Failed to load {self.output_tracking}')
                return ERROR_CODE['FailedToExport'], 'Failed'

            if hasattr(self, 'tracker_out_dir'):
                src_folder = self.tracker_out_dir
                logger.info(f"Using tracker output directory: {src_folder}")
            else:
                src_folder = Path(self.output_tracking)

            tgt_folder = Path(self.output_export)
            config_data = self.tracking_config
            nerf_writer = NeRFDatasetWriter(config_data.data, tgt_folder, None,
                                            None, 'white', device_id=self.device_id)
            nerf_writer.write()
            flame_writer = TrackedFLAMEDatasetWriter(config_data.model,
                                                     src_folder,
                                                     tgt_folder,
                                                     mode='param',
                                                     epoch=-1,
                                                     device_id=self.device_id)
            flame_writer.write()
            split_json(tgt_folder)
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f'Error in export: {e}')
            torch.cuda.empty_cache()
            return ERROR_CODE['FailedToExport'], 'Failed'

   
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    video_input_path ='/inspire/hdd/project/powersystem/fengkairui-25026/wuyue/data/nersemble_data/nersemble_data/sequence_EXP-5-mouth_part-4/237/EXP-5-mouth/cam_222200037.mp4' 
    result_save_path = "/inspire/hdd/project/powersystem/fengkairui-25026/wuyue/project/FastAvatarpp_train/trail"
    gpu_id = 0
    image_size = 512
    target_fps = 3
    head_detect_freq = 4

    tracker = TrackingVideoImage(result_save_path, gpu_id)
    
    tracker.tracking_video(video_input_path, 
                            image_size=image_size,
                            target_fps=target_fps,
                            head_detect_freq=head_detect_freq)