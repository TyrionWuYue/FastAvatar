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
import trimesh
import numpy as np
from PIL import Image
from glob import glob
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from safetensors.torch import load_file
from accelerate.logging import get_logger
from collections import defaultdict
import cv2

from FastAvatar.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
from FastAvatar.runners.infer.base_inferrer import Inferrer
from FastAvatar.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from FastAvatar.runners import REGISTRY_RUNNERS
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

        cfg.save_tmp_dump = os.path.join("exps", 'save_tmp', _relative_path)
        cfg.image_dump = os.path.join("exps", 'images', _relative_path)
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)
        
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, 'images'))
        cfg.setdefault('video_dump', os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join("dumps", cli_cfg.model_name, 'meshes'))
    
    cfg.motion_video_read_fps = 6
    cfg.lora_weights_path = 'lora_001200.safetensors'
    cfg.merge_with(cli_cfg)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')
    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register(name="infer.fast_avatar")
class FastAvatarInferrer(Inferrer):

    EXP_TYPE = "fast_avatar"

    def __init__(self):
        super().__init__()
        
        self.cfg = parse_configs()
        self.model: FastAvatarInferrer = self._build_model(self.cfg).to(self.device)
        self.flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
                                             vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
                                             facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True,
                                             args = self.cfg)

    def _build_model(self, cfg):
        from FastAvatar.models.modeling_fastavatar import ModelFastAvatar
        
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
        
        # Load LoRA weights if specified
        if hasattr(cfg, 'lora_weights_path') and cfg.lora_weights_path is not None:
            lora_path = os.path.join(cfg.model_name, cfg.lora_weights_path)
            print(f"\nLoading LoRA weights from {lora_path}")
            if lora_path.endswith('.safetensors'):
                lora_ckpt = load_file(lora_path, device='cpu')
            else:
                lora_ckpt = torch.load(lora_path, map_location='cpu')
            
            # Load LoRA weights
            model.transformer.frame_attn.load_state_dict(lora_ckpt, strict=False)
            print("Finished loading LoRA weights from:", lora_path)
        
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

    def infer(self):
        # Get input image paths
        if os.path.isfile(self.cfg.image_input):
            image_paths = [self.cfg.image_input]
            omit_prefix = os.path.dirname(self.cfg.image_input)
        else:
            image_paths = glob(os.path.join(self.cfg.image_input, "*.png"))
            omit_prefix = self.cfg.image_input

        # Initialize lists to store processed data
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        flame_params = defaultdict(list)

        # Process each image
        print(f"\nProcessing {len(image_paths)} images...")
        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):
            try:
                print(f"\nProcessing image: {image_path}")
                # Preprocess input image
                return_code = self.flametracking.preprocess(image_path)
                assert (return_code == 0), "flametracking preprocess failed!"
                return_code = self.flametracking.optimize()
                assert (return_code == 0), "flametracking optimize failed!"
                return_code, output_dir = self.flametracking.export()
                assert (return_code == 0), "flametracking export failed!"

                image_path = os.path.join(output_dir, "images/00000_00.png")
                mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
                print(f"Processed image saved to: {image_path}")
                
                # Process image and get FLAME parameters
                image, mask, intr, shape_param = preprocess_image(
                    image_path,
                    mask_path=mask_path,
                    intr=None,
                    pad_ratio=0,
                    bg_color=1.0,
                    max_tgt_size=None,
                    aspect_standard=1.0,
                    enlarge_ratio=[1.0, 1.0],
                    render_tgt_size=self.cfg.source_size,
                    multiply=14,
                    need_mask=True,
                    get_shape_param=True
                )
                print(f"Image shape after preprocessing: {image.shape}")

                # Collect data
                
                rgbs.append(image)  # Already has batch dim
                masks.append(mask)  # Already has batch dim

                
            except Exception as e:
                print(f"Error processing image {image_path}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                traceback.print_exc()
        
        # Prepare motion sequences
        print(f"Preparing motion sequences from: {self.cfg.motion_seqs_dir}")
        motion_seqs = prepare_motion_seqs(
            motion_seqs_dir=self.cfg.motion_seqs_dir,
            image_folder=self.cfg.motion_img_dir,
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
        intrs.append(motion_seqs["intrs"])  # Already has batch dim
        bg_colors.append(motion_seqs["bg_colors"])  # Already has batch dim

        # Collect FLAME parameters
        for k, v in motion_seqs["flame_params"].items():
            flame_params[k].append(v)  # Already has batch dim
            
        # Stack all collected data (maintaining batch dimension)
        c2ws = torch.cat(c2ws, dim=1)  # [1, N, 4, 4]
        intrs = torch.cat(intrs, dim=1)  # [1, N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        rgbs = rgbs.unsqueeze(0)  # [1, N, 3, H, W]
        bg_colors = torch.cat(bg_colors, dim=1)
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
        masks = masks.unsqueeze(0)  # [1, N, 1, H, W]

        # Stack FLAME parameters (maintaining batch dimension)
        for k in flame_params:
            flame_params[k] = torch.cat(flame_params[k], dim=1)  # [1, N, ...]
        
        n_inf = c2ws.shape[1]
        flame_params["betas"] = flame_params["betas"].unsqueeze(0)

        # Move all tensors to GPU
        device = self.device
        rgbs = rgbs.to(device)
        c2ws = c2ws.to(device)
        intrs = intrs.to(device)
        bg_colors = bg_colors.to(device)
        masks = masks.to(device)
        flame_params = {k: v.to(device) for k, v in flame_params.items()}

        # Run model inference
        print("\nStarting model inference.........................")
        start_time = time.time()

        with torch.no_grad():
            frame_results = []
            if not os.path.isfile(self.cfg.image_input) and self.cfg.get("if_multi_frames_compare", False):
                for i in range(rgbs.shape[1]):
                    res = self.model.infer_iamges(
                        rgbs[:, i:i+1],
                        c2ws,
                        intrs,
                        bg_colors,
                        flame_params=flame_params
                    )
                    frame_results.append(res["comp_rgb"].detach().cpu().numpy())
                
            # inference for all frames
            res = self.model.infer_iamges(
                rgbs,
                c2ws,
                intrs,
                bg_colors,
                flame_params=flame_params
            )
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
        
        print(f"Inference completed in {time.time() - start_time:.2f} seconds")

        # Process results
        print("\nProcessing inference results...")
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        only_pred = rgb

        if self.cfg.get("vis_motion", False):
            # vis_ref_img = np.tile(cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            interpolation = get_smart_interpolation(vis_ref_img.shape[:2], (rgb[0].shape[0], rgb[0].shape[1]))
            vis_ref_img = np.tile(cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=interpolation)[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            blend_ratio = 0.7
            blend_res = ((1 - blend_ratio) * rgb + blend_ratio * motion_seqs["vis_motion_render"]).astype(np.uint8)
            rgb = np.concatenate([vis_ref_img, rgb, motion_seqs["vis_motion_render"]], axis=2)

        # Save results
        print("\nSaving results...")
        if os.path.isfile(self.cfg.image_input):
            # For single image input, use the original logic
            uid = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        else:
            # For folder input, use the folder name
            uid = os.path.basename(self.cfg.image_input)
        
        dump_video_path = os.path.join(self.cfg.video_dump, f'{uid}.mp4')
        dump_image_dir = os.path.join(self.cfg.image_dump, f'{uid}')
        dump_tmp_dir = os.path.join(self.cfg.image_dump, "tmp_res")
        dump_mesh_path = os.path.join(self.cfg.mesh_dump)
        
        os.makedirs(dump_image_dir, exist_ok=True)
        os.makedirs(dump_tmp_dir, exist_ok=True)
        os.makedirs(dump_mesh_path, exist_ok=True)

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

        # Save individual frames if requested
        if self.cfg.get("save_img", False) and dump_image_dir is not None:
            print("\nSaving individual frames...")
            for i in range(rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(only_pred[i]).save(save_file)

                if self.cfg.get("save_ply", False) and dump_mesh_path is not None:
                    res["3dgs"][i][0][0].save_ply(os.path.join(dump_image_dir, f"{i:04d}.ply"))

            # Save canonical mesh
            dump_cano_dir = "./exps/cano_gs/"
            os.makedirs(dump_cano_dir, exist_ok=True)
            
            # Save canonical point cloud
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + "_gs_offset.ply")
            res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=False, offset2xyz=True)
            print(f"Canonical point cloud saved to: {cano_ply_pth}")

            # Save canonical mesh
            import trimesh
            vtxs = res['cano_gs_lst'][0].xyz - res['cano_gs_lst'][0].offset
            vtxs = vtxs.detach().cpu().numpy()
            faces = self.model.renderer.flame_model.faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vtxs, faces=faces)
            mesh.export(os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_shaped_mesh.obj'))
            print(f"Canonical mesh saved to: {os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_shaped_mesh.obj')}")

            # Save textured mesh
            import FastAvatar.models.rendering.utils.mesh_utils as mesh_utils
            vtxs = res['cano_gs_lst'][0].xyz.detach().cpu()
            faces = self.model.renderer.flame_model.faces.detach().cpu()
            colors = res['cano_gs_lst'][0].shs.squeeze(1).detach().cpu()
            pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_textured_mesh.obj')
            mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")
            print(f"Textured mesh saved to: {pth}")

        print("\nInference and saving completed successfully!")

def get_smart_interpolation(src_size, dst_size):
    if src_size[0] < dst_size[0] or src_size[1] < dst_size[1]:
        return cv2.INTER_CUBIC
    else:
        return cv2.INTER_AREA