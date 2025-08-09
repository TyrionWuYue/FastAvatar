import argparse
import json
import os
import time
from pathlib import Path
import glob
import random

import cv2
import numpy as np
import torch
import torchvision
import tyro
import yaml
from loguru import logger
from PIL import Image

from external.human_matting import StyleMatteEngine as HumanMattingEngine
from external.landmark_detection.FaceBoxesV2.faceboxes_detector import \
    FaceBoxesDetector
from external.landmark_detection.infer_folder import Alignment
from external.vgghead_detector import VGGHeadDetector
from vhap.config.base import BaseTrackingConfig
from vhap.export_as_nerf_dataset import (NeRFDatasetWriter,
                                         TrackedFLAMEDatasetWriter, split_json)
from vhap.model.tracker import GlobalTracker

# Define error codes for various processing failures.
ERROR_CODE = {'FailedToDetect': 1, 'FailedToOptimize': 2, 'FailedToExport': 3}


def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    """Calculate new target size that is divisible by multiply.
    
    Args:
        cur_hw: Current height and width tuple (h, w)
        aspect_standard: Target aspect ratio (h/w)
        tgt_size: Target size
        multiply: Number that the final size should be divisible by
        
    Returns:
        tuple: (new_height, new_width), ratio_y, ratio_x
    """
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_size = tgt_size * aspect_standard, tgt_size
    tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def process_data_augmentation(frame_dir, aspect_standard=1.0, render_tgt_size=512, multiply=14):
    """Apply data augmentation to get final dataset-ready data."""
    try:
        # Debug: Log the actual frame_dir value
        logger.info(f'process_data_augmentation called with frame_dir: {frame_dir}')
        
        # Load the processed images from export directory
        images_dir = os.path.join(frame_dir, 'export', 'images')
        if not os.path.exists(images_dir):
            logger.warning(f'Images directory not found: {images_dir}')
            return False
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        if not image_files:
            logger.warning(f'No image files found in {images_dir}')
            return False
        
        # Landmark2d data is stored in the preprocess directory
        landmark_dir = os.path.join(frame_dir, 'preprocess', 'landmark2d')
        if not os.path.exists(landmark_dir):
            logger.warning(f'Landmark2d directory not found: {landmark_dir}')
            return False
        
        # For video processing, we use the landmarks.npz file
        landmark_path = os.path.join(landmark_dir, 'landmarks.npz')
        if not os.path.exists(landmark_path):
            logger.warning(f'Landmarks file not found: {landmark_path}')
            return False
        
        transforms_path = os.path.join(frame_dir, 'export', 'transforms.json')
        if not os.path.exists(transforms_path):
            logger.warning(f'Transforms.json not found: {transforms_path}')
            return False
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
               
        
        processed_data_dir = os.path.join(os.path.dirname(frame_dir), 'processed_data')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Process each image using the corresponding landmarks
        for frame_idx, img_path in enumerate(image_files):
            try:
                logger.info(f'Processing frame {frame_idx + 1}/{len(image_files)}')
                
                # Create frame-specific directory
                frame_dir_name = f'{frame_idx:05d}'
                frame_save_dir = os.path.join(processed_data_dir, frame_dir_name)
                os.makedirs(frame_save_dir, exist_ok=True)
                
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
                
                # 2. Process mask
                mask_path = os.path.join(frame_dir, 'preprocess', 'mask', f'{frame_idx:05d}.png')
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path))
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]
                else:
                    raise FileNotFoundError(f'Mask file not found: {mask_path}')
                
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                mask = (mask > 0.5).astype(np.float32)
                
                # 3. Apply background color (fixed white for inference)
                bg_color = 1.0  # Fixed white background
                
                # 4. Resize to render_tgt_size for training (no cropping to preserve intrinsics)
                current_h, current_w = rgb.shape[:2]
                logger.info(f'Current image size: {current_h}x{current_w}, target size: {render_tgt_size}')
                
                # Calculate resize ratio
                ratio = render_tgt_size / max(current_h, current_w)
                new_h = int(current_h * ratio)
                new_w = int(current_w * ratio)
                
                # Ensure the new size is divisible by multiply
                new_h = (new_h // multiply) * multiply
                new_w = (new_w // multiply) * multiply
                
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
                rgb_resized = torchvision.transforms.functional.resize(rgb_tensor, (new_h, new_w), antialias=True)
                mask = cv2.resize(mask, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                
                rgb = rgb_resized.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

                # Update ratios for intrinsic matrix
                ratio_y = new_h / current_h
                ratio_x = new_w / current_w
                
                # Update intrinsic matrix
                intr[0, 0] *= ratio_x
                intr[1, 1] *= ratio_y
                intr[0, 2] *= ratio_x
                intr[1, 2] *= ratio_y
                
                # Ensure RGB values are in [0, 1] range after resize
                rgb = np.clip(rgb, 0.0, 1.0)
                mask = np.clip(mask, 0.0, 1.0)
                
                # Convert to torch tensors
                rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)  # [3, H, W]
                mask = torch.from_numpy(mask).float().unsqueeze(0)    # [1, H, W]
                
                # Save processed data
                np.save(os.path.join(frame_save_dir, 'rgb.npy'), rgb.numpy())
                np.save(os.path.join(frame_save_dir, 'mask.npy'), mask.numpy())
                np.save(os.path.join(frame_save_dir, 'intrs.npy'), intr.numpy())
                np.save(os.path.join(frame_save_dir, 'bg_color.npy'), np.array(bg_color))
                
                logger.info(f'Frame {frame_idx} processed and saved to {frame_save_dir}')
                
            except Exception as e:
                logger.error(f'Error processing frame {frame_idx}: {e}')
                continue
        
        logger.info(f'Data augmentation completed for {len(image_files)} frames')
        return True
        
    except Exception as e:
        logger.error(f'Error in process_data_augmentation: {e}')
        return False


def expand_bbox(bbox, scale=1.1):
    """Expands the bounding box by a given scale."""
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    extension_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
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
    """Run landmark detection on all images in a folder."""
    try:
        # Get all image files
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
        logger.info(f'Found {len(image_files)} images for landmark detection')
        
        # Create output directory
        landmark_output_dir = os.path.join(output_dir, 'landmark2d')
        os.makedirs(landmark_output_dir, exist_ok=True)
        
        # Load face detector
        facebox_model_path = './model_zoo/flame_tracking_models/FaceBoxesV2.pth'
        detector = FaceBoxesDetector('FaceBoxes', facebox_model_path, True, f'cuda:{device_id}')
        
        # Load alignment model
        alignment_model_path = './model_zoo/flame_tracking_models/68_keypoints_model.pkl'
        args = argparse.Namespace()
        args.config_name = "alignment"
        args.model_path = alignment_model_path
        alignment = Alignment(args, alignment_model_path, dl_framework='pytorch', device_ids=[device_id])
        
        # Process each image
        keypoint_data = {}
        
        for frame_index, image_path in enumerate(image_files):
            try:
                logger.info(f'Detecting landmarks for image {frame_index + 1}/{len(image_files)}')
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f'Failed to load image: {image_path}')
                    continue
                
                # Detect faces
                detections, _ = detector.detect(image, 0.8, 1)
                if not detections:
                    logger.warning(f'No faces detected in {image_path}')
                    continue
                
                # Use the first detected face
                detection = detections[0]
                x1_ori, y1_ori = detection[2], detection[3]
                x2_ori, y2_ori = x1_ori + detection[4], y1_ori + detection[5]
                
                scale = max(x2_ori - x1_ori, y2_ori - y1_ori) / 180
                center_w, center_h = (x1_ori + x2_ori) / 2, (y1_ori + y2_ori) / 2
                scale, center_w, center_h = float(scale), float(center_w), float(center_h)
                
                # Detect landmarks
                face_landmarks = alignment.analyze(image, scale, center_w, center_h)
                
                # Store landmarks
                frame_name = os.path.basename(image_path)
                keypoint_data[frame_name] = face_landmarks.tolist()
                
            except Exception as e:
                logger.error(f'Error detecting landmarks for {image_path}: {e}')
                continue
        
        # Save keypoint data
        if keypoint_data:
            keypoint_path = os.path.join(output_dir, 'keypoint.json')
            with open(keypoint_path, 'w') as f:
                json.dump(keypoint_data, f, indent=2)
            logger.info(f'Saved keypoint data to {keypoint_path}')
            return True
        else:
            logger.error('No keypoint data generated')
            return False
            
    except Exception as e:
        logger.error(f'Error in landmark detection: {e}')
        return False


class FlameTrackingMultiImage:
    """Class for tracking and processing multiple images from a folder."""
    
    def __init__(
            self,
            output_dir,
            alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
            vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
            human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
            facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
            detect_iris_landmarks=False,
            args=None):

        logger.info(f'Output Directory: {output_dir}')

        start_time = time.time()
        logger.info('Loading Pre-trained Models...')

        self.output_dir = output_dir
        self.output_preprocess = os.path.join(output_dir, 'preprocess')
        self.output_tracking = os.path.join(output_dir, 'tracking')
        self.output_export = os.path.join(output_dir, 'export')
        self.device = 'cuda:0'

        # Load alignment model
        assert os.path.exists(
            alignment_model_path), f'{alignment_model_path} does not exist!'
        if args is None:
            args = self._parse_args()
        args.config_name = "alignment"
        args.model_path = alignment_model_path
        self.alignment = Alignment(args,
                                   alignment_model_path,
                                   dl_framework='pytorch',
                                   device_ids=[0])

        # Load VGG head model
        assert os.path.exists(
            vgghead_model_path), f'{vgghead_model_path} does not exist!'
        self.vgghead_encoder = VGGHeadDetector(
            device=self.device, vggheadmodel_path=vgghead_model_path)

        # Load human matting model
        assert os.path.exists(
            human_matting_path), f'{human_matting_path} does not exist!'
        self.matting_engine = HumanMattingEngine(
            device=self.device, human_matting_path=human_matting_path)

        # Load face box detector model
        assert os.path.exists(
            facebox_model_path), f'{facebox_model_path} does not exist!'
        self.detector = FaceBoxesDetector('FaceBoxes', facebox_model_path,
                                          True, self.device)

        self.detect_iris_landmarks_flag = detect_iris_landmarks
        if self.detect_iris_landmarks_flag:
            from fdlite import FaceDetection, FaceLandmark, IrisLandmark
            self.iris_detect_faces = FaceDetection()
            self.iris_detect_face_landmarks = FaceLandmark()
            self.iris_detect_iris_landmarks = IrisLandmark()

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(f'Finished Loading Pre-trained Models. Time: '
                    f'{end_time - start_time:.2f}s')

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='Evaluation script')
        parser.add_argument('--output_dir',
                            type=str,
                            help='Output directory',
                            default='output')
        parser.add_argument('--config_name',
                            type=str,
                            help='Configuration name',
                            default='alignment')
        parser.add_argument('--blender_path',
                            type=str,
                            help='Blender path')
        return parser.parse_args()

    def preprocess(self, input_folder, max_frames=None):
        """Preprocess images in the input folder for tracking."""
        if not os.path.exists(input_folder):
            logger.warning(f'{input_folder} does not exist!')
            return ERROR_CODE['FailedToDetect']

        start_time = time.time()
        logger.info('Starting Preprocessing...')
        
        # Get all image files from the input folder
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        if not image_files:
            logger.error(f'No image files found in {input_folder}')
            return ERROR_CODE['FailedToDetect']
        
        # Sort image files to ensure consistent ordering
        image_files = sorted(image_files)
        
        # Limit the number of frames if max_frames is specified
        original_count = len(image_files)
        if max_frames is not None and len(image_files) > max_frames:
            image_files = image_files[:max_frames]
            logger.info(f'Limited to {max_frames} frames (out of {original_count} total)')
        else:
            logger.info(f'Found {len(image_files)} images to process')
        
        # Create output directories
        output_image_dir = os.path.join(self.output_preprocess, 'images')
        output_mask_dir = os.path.join(self.output_preprocess, 'mask')
        output_alpha_map_dir = os.path.join(self.output_preprocess, 'alpha_maps')
        landmark_output_dir = os.path.join(self.output_preprocess, 'landmark2d')

        # Clean existing directories to ensure we only have the frames we want
        import shutil
        for dir_path in [output_image_dir, output_mask_dir, output_alpha_map_dir, landmark_output_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f'Cleaned existing directory: {dir_path}')

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_alpha_map_dir, exist_ok=True)
        os.makedirs(landmark_output_dir, exist_ok=True)

        # Step 1: Process all images for head detection and matting
        logger.info('Step 1: Processing images for head detection and matting...')
        processed_images = []
        
        for frame_index, image_path in enumerate(image_files):
            try:
                logger.info(f'Processing image {frame_index + 1}/{len(image_files)}: {os.path.basename(image_path)}')
                
                # Load the original image
                frame = torchvision.io.read_image(image_path)[:3, ...]

                # Head detection and bounding box calculation
                try:
                    _, frame_bbox, _ = self.vgghead_encoder(frame, frame_index)
                except Exception:
                    logger.error(f'Failed to detect head in {image_path}')
                    continue

                if frame_bbox is None:
                    logger.error(f'Failed to detect head in {image_path}')
                    continue
                
                # Expand bounding box
                frame_bbox = expand_bbox(frame_bbox, scale=1.65).long()

                # Crop and resize
                cropped_frame = torchvision.transforms.functional.crop(
                    frame,
                    top=frame_bbox[1],
                    left=frame_bbox[0],
                    height=frame_bbox[3] - frame_bbox[1],
                    width=frame_bbox[2] - frame_bbox[0])
                cropped_frame = torchvision.transforms.functional.resize(
                    cropped_frame, (1024, 1024), antialias=True)

                # Apply matting
                cropped_frame, mask = self.matting_engine(cropped_frame / 255.0,
                                                          return_type='matting',
                                                          background_rgb=1.0)
                cropped_frame = cropped_frame.cpu() * 255.0
                saved_image = np.round(cropped_frame.cpu().permute(
                    1, 2, 0).numpy()).astype(np.uint8)[:, :, (2, 1, 0)]

                # Save processed image, mask and alpha map
                frame_name = f'{frame_index:05d}.png'
                cv2.imwrite(os.path.join(output_image_dir, frame_name), saved_image)
                cv2.imwrite(os.path.join(output_mask_dir, frame_name),
                            np.array((mask.cpu() * 255.0)).astype(np.uint8))
                cv2.imwrite(
                    os.path.join(output_alpha_map_dir, frame_name.replace('.png', '.jpg')),
                    (np.ones_like(saved_image) * 255).astype(np.uint8))
                
                processed_images.append((frame_index, saved_image))
                
            except Exception as e:
                logger.error(f'Error processing image {image_path}: {e}')
                continue

        if not processed_images:
            logger.error('No images were successfully processed')
            return ERROR_CODE['FailedToDetect']

        # Step 2: Run batch landmark detection on all processed images
        logger.info('Step 2: Running batch landmark detection...')
        if not run_landmark_detection_on_folder(output_image_dir, self.output_preprocess, 0):
            logger.error('Failed to run batch landmark detection')
            return ERROR_CODE['FailedToDetect']

        # Step 3: Convert landmark results to the expected format
        logger.info('Step 3: Converting landmark results...')
        keypoint_json_path = os.path.join(self.output_preprocess, 'keypoint.json')
        if os.path.exists(keypoint_json_path):
            with open(keypoint_json_path, 'r') as f:
                keypoint_data = json.load(f)
            
            # For multi-image processing, we need to create landmark data for all frames
            if processed_images and keypoint_data:
                # Collect all landmarks for all frames
                all_landmarks = []
                all_bounding_boxes = []
                
                for frame_index, _ in processed_images:
                    frame_filename = f'{frame_index:05d}.png'
                    if frame_filename in keypoint_data:
                        landmarks = keypoint_data[frame_filename]
                        
                        # Normalize landmarks for landmark2d format
                        normalized_landmarks = np.zeros((len(landmarks), 3))
                        normalized_landmarks[:, :2] = np.array(landmarks) / 1024
                        
                        all_landmarks.append(normalized_landmarks)
                        all_bounding_boxes.append([])  # Empty bounding box for compatibility
                    else:
                        logger.warning(f'No landmark data found for frame {frame_filename}')
                        # Use first frame's landmarks as fallback
                        if all_landmarks:
                            all_landmarks.append(all_landmarks[0].copy())
                            all_bounding_boxes.append([])
                        else:
                            logger.error(f'No landmark data available for frame {frame_filename}')
                            continue
                
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

                    # Save landmark data
                    landmark_path = os.path.join(landmark_output_dir, 'landmarks.npz')
                    np.savez(landmark_path, **landmark_data)
                    logger.info(f'Successfully saved landmark2d data for {len(all_landmarks)} frames to {landmark_path}')
                else:
                    logger.error('No landmark data available for any frame')
            else:
                logger.warning(f'No processed images or landmark data available for landmark conversion')

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(
            f'Finished Processing Images. Time: {end_time - start_time:.2f}s')

        return 0

    def optimize(self):
        """Optimize the tracking model using configuration data."""
        start_time = time.time()
        logger.info('Starting Optimization...')

        try:
            tyro.extras.set_accent_color('bright_yellow')
            from yaml import safe_load, safe_dump
            with open("configs/vhap_tracking/base_tracking_config.yaml", 'r') as yml_f:
                config_data = safe_load(yml_f)
            config_data = tyro.from_yaml(BaseTrackingConfig, config_data)

            # Use the preprocess directory as the sequence directory
            config_data.data.sequence = os.path.basename(self.output_preprocess)
            # Use the directory containing preprocess as root folder
            config_data.data.root_folder = Path(os.path.dirname(self.output_preprocess))

            if not os.path.exists(self.output_preprocess):
                logger.error(f'Failed to load {self.output_preprocess}')
                return ERROR_CODE['FailedToOptimize']

            config_data.exp.output_folder = Path(self.output_tracking)
            tracker = GlobalTracker(config_data)
            tracker.optimize()

            end_time = time.time()
            torch.cuda.empty_cache()
            logger.info(
                f'Finished Optimization. Time: {end_time - start_time:.2f}s')

            return 0
        except Exception as e:
            logger.error(f'Error in optimization: {e}')
            torch.cuda.empty_cache()
            return ERROR_CODE['FailedToOptimize']

    def export(self):
        """Export the tracking results to configured folder."""
        logger.info(f'Beginning export from {self.output_tracking}')
        start_time = time.time()
        
        try:
            if not os.path.exists(self.output_tracking):
                logger.error(f'Failed to load {self.output_tracking}')
                return ERROR_CODE['FailedToExport'], 'Failed'

            src_folder = Path(self.output_tracking)
            tgt_folder = Path(self.output_export)
            src_folder, config_data = load_config(src_folder)

            nerf_writer = NeRFDatasetWriter(config_data.data, tgt_folder, None,
                                            None, 'white')
            nerf_writer.write()

            flame_writer = TrackedFLAMEDatasetWriter(config_data.model,
                                                     src_folder,
                                                     tgt_folder,
                                                     mode='param',
                                                     epoch=-1)
            flame_writer.write()

            split_json(tgt_folder)

            # Apply data augmentation to get final dataset-ready data
            logger.info('Applying data augmentation...')
            aspect_standard = 1.0
            render_tgt_size = 512
            multiply = 14
            
            if hasattr(self, 'args') and self.args is not None:
                if hasattr(self.args, 'source_size'):
                    render_tgt_size = self.args.source_size
                if hasattr(self.args, 'render_size'):
                    render_tgt_size = self.args.render_size
            
            # Pass the output directory (parent of export) so process_data_augmentation can find preprocess
            # and create processed_data at the same level as export
            process_data_augmentation(str(self.output_dir), aspect_standard, render_tgt_size, multiply)

            end_time = time.time()
            torch.cuda.empty_cache()
            logger.info(f'Finished Export. Time: {end_time - start_time:.2f}s')

            return 0, str(tgt_folder)
        except Exception as e:
            logger.error(f'Error in export: {e}')
            torch.cuda.empty_cache()
            return ERROR_CODE['FailedToExport'], 'Failed' 