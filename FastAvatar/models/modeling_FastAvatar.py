# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
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
import time
import math
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from FastAvatar.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from FastAvatar.models.alternating_cross_attn import AlternatingCrossAttn
from diffusers.utils import is_torch_version
from FastAvatar.models.heads.dpt_head import DPTHead
from FastAvatar.models.track_head import TrackHead

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFastAvatar(nn.Module):
    def __init__(self,
                 transformer_dim: int = 1024, 
                 transformer_layers: int = 10,
                 transformer_heads: int = 16,
                 aa_order: list = ["frame", "global"],
                 aa_block_size: int = 1,
                 tf_grad_ckpt=True,
                 encoder_grad_ckpt=True,
                 encoder_freeze: bool = True, encoder_type: str = 'dinov2_fusion',
                 encoder_model_name: str = 'dinov2_vitl14_reg',
                 encoder_feat_dim: int = 1024,
                 pcl_dim: int=1024,
                 human_model_path="./model_zoo/human_parametric_models",
                 flame_subdivide_num=1,
                 flame_type="flame",
                 gs_query_dim=1024,
                 gs_use_rgb=True,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=0.2,
                 gs_clip_scaling=0.01,
                 fix_opacity=False,
                 fix_rotation=False,
                 use_flame_tokens: bool = True,
                 flame_encoder_config: dict = None,
                 use_camera_tokens: bool = True,
                 camera_encoder_config: dict = None,
                 use_multi_frame_pc: bool = True,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.use_multi_frame_pc = use_multi_frame_pc
        self.conf_loss_frames = kwargs.get("conf_loss_frames", 4)  # Number of frames for confidence loss

        # image encoder
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=self.encoder_feat_dim,
        )

        # learnable points embedding
        self.pcl_embed = PointEmbed(dim=pcl_dim)

        # renderer
        self.num_sliced_frames = kwargs.get("num_sliced_frames", 2)
        self.renderer = GS3DRenderer(human_model_path=human_model_path,
                                     subdivide_num=flame_subdivide_num,
                                     smpl_type=flame_type,
                                     feat_dim=transformer_dim,
                                     query_dim=gs_query_dim,
                                     use_rgb=gs_use_rgb,
                                     sh_degree=gs_sh,
                                     mlp_network_config=gs_mlp_network_config,
                                     xyz_offset_max_step=gs_xyz_offset_max_step,
                                     clip_scaling=gs_clip_scaling,
                                     scale_sphere=kwargs.get("scale_sphere", False),
                                     fix_opacity=fix_opacity,
                                     fix_rotation=fix_rotation,
                                     skip_decoder=True,
                                     decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
                                     gradient_checkpointing=self.gradient_checkpointing,
                                     add_teeth=kwargs.get("add_teeth", False),
                                     teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
                                     oral_mesh_flag=kwargs.get("oral_mesh_flag", False),
                                     use_mesh_shading=kwargs.get('use_mesh_shading', False),
                                     render_rgb=kwargs.get("render_rgb", True),
                                     )
        
        self.patch_start_idx = 0
        if use_flame_tokens:
            self.patch_start_idx += 1
        if use_camera_tokens:
            self.patch_start_idx += 1

        # Initialize transformer parameters
        self.intermediate_layer_idx = kwargs.get("intermediate_layer_idx", [2,5,8])

        self.transformer = AlternatingCrossAttn(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=transformer_dim,
            gradient_checkpointing=self.gradient_checkpointing,
            aa_order=aa_order,
            aa_block_size=aa_block_size,
            intermediate_layer_idx=self.intermediate_layer_idx,
            use_flame_tokens=use_flame_tokens,
            flame_encoder_config=flame_encoder_config,
            use_camera_tokens=use_camera_tokens,
            camera_encoder_config=camera_encoder_config,
            patch_start_idx=self.patch_start_idx
        )
        
        # Track Head
        self.use_tracking = kwargs.get("use_tracking", True)
        self.use_confidence = kwargs.get("use_confidence", True)
        
        if self.use_tracking:
            self.track_head = TrackHead(
                dim_in = transformer_dim * 2, 
                intermediate_layer_idx=self.intermediate_layer_idx,
                use_flame_tokens=use_flame_tokens
            )
        else:
            self.track_head = None
            logger.info("Tracking head disabled")

        # confidence head
        if self.use_confidence:
            self.confidence_head = DPTHead(dim_in=transformer_dim*2, output_dim=1, activation='inv_log', conf_activation='expp1')
        else:
            self.confidence_head = None
            logger.info("Confidence head disabled")

    @staticmethod
    def _encoder_fn(encoder_type: str):
        from FastAvatar.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
        return Dinov2FusionWrapper

    def forward_encode_image(self, image):
        """
        Encode image features, supporting both single and multi-frame inputs
        Args:
            image: [B*N_frames, C_img, H_img, W_img]
        Returns:
            image_feats: [B*N_frames, H*W, C]
        """
        B, N_frames, C_img, H_img, W_img = image.shape
        image = image.view(B * N_frames, C_img, H_img, W_img)
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image)
        return image_feats
    
    def forward_transformer(self, image_feats, camera_embeddings, query_points, query_feats=None, flame_params=None, camera_params=None):
        """
        Args:
            image_feats: [B, N_input, H*W, C]
            query_points: [B, N_points, 3]
            flame_params: Dictionary containing FLAME parameters for encoding into tokens
            camera_params: Dictionary containing camera parameters for encoding into tokens
        Returns:
            latent_points: [B, N_input, N_points, C]
            intermediate_outputs: List of intermediate layer outputs
        """
        B, N_input = image_feats.shape[:2]
        B2 = query_points.shape[0]
        assert B == B2, "Batch size must match"

        # Reshape query_points for pcl_embed
        query_points_reshape = query_points.reshape(B, -1, 3)  # [B*N_input, N_points, 3]
        x = self.pcl_embed(query_points_reshape)  # [B*N_input, N_points, C]
        x = x.reshape(B, N_input, -1, x.shape[-1])  # [B, N_input, N_points, C]
        x = x.to(image_feats.dtype)
        if query_feats is not None:
            x = x + query_feats.to(image_feats.dtype)
        
        x, cond, intermediate_outputs = self.transformer(x, image_feats, mod=camera_embeddings, flame_params=flame_params, camera_params=camera_params)
        
        return x, cond, intermediate_outputs

    @torch.compile
    def forward_latent_points(self, image, query_points=None, flame_params=None, camera_params=None):
        """
        Process image and query points to get latent features
        Args:
            image: [B, N_input, C_img, H_img, W_img] - N_input: Output frame number
            query_points: [B, N_points, 3]
            flame_params: Dictionary containing FLAME parameters for encoding into tokens
            camera_params: Dictionary containing camera parameters for encoding into tokens
        Returns:
            points_embedding: [B, N_inf, N_points, C]
            image_feats: [B, N_input, H*W, C]
            intermediate_outputs: List of intermediate layer outputs
        """
        B, N_input = image.shape[:2]
        B2, N_input2 = query_points.shape[:2]
        assert B == B2, "Batch size must match"
        assert N_input == N_input2, "Number of Frames must match"
        
        # encode image
        image_feats = self.forward_encode_image(image)
        
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # Convert query_points to the same dtype as image_feats
        query_points = query_points.to(image_feats.dtype)
        image_feats = image_feats.view(B, N_input, *image_feats.shape[1:])
        # Get transformer output
        latent_points, cond, intermediate_outputs = self.forward_transformer(image_feats, camera_embeddings=None, query_points=query_points, flame_params=flame_params, camera_params=camera_params)

        return latent_points, intermediate_outputs

    def _render_multiple_frames(self, latent_points, query_points, inf_flame_params, c2ws, intrs, bg_colors, render_h, render_w, N_inf, input_indices=[0]):
        """
        Render multiple frames using the same latent points and query points.
        
        Args:
            latent_points: [B, N_points, C] - Latent features for 3DGS
            query_points: [B, N_points, 3] - 3D query points
            inf_flame_params: Dict containing FLAME parameters for all frames
            c2ws: [B, N_inf, 4, 4] - Camera to world transformations
            intrs: [B, N_inf, 4, 4] - Camera intrinsics
            bg_colors: [B, N_inf, 3] - Background colors
            render_h, render_w: int - Render resolution
            N_inf: int - Number of frames to render

            rotation: [B, N_inf, 3] - Rotation parameters
            translation: [B, N_inf, 3] - Translation parameters
            expr: [B, N_inf, 100] - Expression parameters
            neck_pose: [B, N_inf, 3] - Neck pose parameters
            jaw_pose: [B, N_inf, 3] - Jaw pose parameters
            eyes_pose: [B, N_inf, 3] - Eyes pose parameters
            
        Returns:
            Dict containing concatenated render results for all frames
        """
        render_res_list = []
        try:
            for f in range(N_inf):
                # Get frame-specific parameters
                frame_flame_params = {}
                for k, v in inf_flame_params.items():
                    if isinstance(v, torch.Tensor):
                        frame_flame_params[k] = v[:, f:f+1]
                    else:
                        frame_flame_params[k] = v

                # Convert all inputs to float32 for rasterization
                latent_points_f32 = latent_points.float()
                query_points_f32 = query_points.float()
                frame_c2ws = c2ws[:, f:f+1].float()
                frame_intrs = intrs[:, f:f+1].float()
                frame_bg_colors = bg_colors[:, f:f+1].float()
                frame_flame_params = {k: v.float() if isinstance(v, torch.Tensor) else v 
                                    for k, v in frame_flame_params.items()}

                # Render this frame
                render_res = self.renderer(
                    gs_hidden_features=latent_points_f32,
                    query_points=query_points_f32,
                    flame_data=frame_flame_params,
                    c2w=frame_c2ws,
                    intrinsic=frame_intrs,
                    height=render_h,
                    width=render_w,
                    background_color=frame_bg_colors,
                    num_input_frames=len(input_indices),
                )
                render_res_list.append(render_res)
                
                # Clean up frame-specific variables
                del render_res, frame_flame_params, latent_points_f32, query_points_f32, frame_c2ws, frame_intrs, frame_bg_colors
                
                # Force memory cleanup after each frame
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine results from all frames
            out = {}  # Changed from defaultdict to regular dict
            for res in render_res_list:
                for k, v in res.items():
                    if k not in out:
                        out[k] = []
                    out[k].append(v)
            
            # Process each key in the output dictionary
            for k, v in out.items():
                if isinstance(v[0], torch.Tensor):
                    out[k] = torch.concat(v, dim=1)
                    if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                        out[k] = out[k].permute(0, 1, 2, 3, 4) # [B, N_inf, H, W, C]
                else:
                    out[k] = v
                    
            return out
        finally:
            # Clean up render results to prevent memory accumulation
            del render_res_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, input_image, target_image, input_c2ws, target_c2ws, input_intrs, target_intrs, input_bg_colors, target_bg_colors, landmarks, input_flame_params, inf_flame_params, uid):
        """
        Multi-frame forward pass using loop for each frame
        Args:
            input_image: [B, N_input, C_img, H_img, W_img]
            target_image: [B, N_target, C_img, H_img, W_img]
            landmarks: [B, N_input, 68, 2]
            input_c2ws: [B, N_input, 4, 4]
            target_c2ws: [B, N_target, 4, 4]
            input_intrs: [B, N_input, 4, 4]
            target_intrs: [B, N_target, 4, 4]
            bg_colors: [B, N_target, 3]
            inf_flame_params: Dict containing FLAME parameters for target frames (N_target)
            input_flame_params: Dict containing FLAME parameters for input frames (N_input)
        Returns:
            Dict containing rendered results and intermediate features
        """
        B, N_input = input_image.shape[:2]
        render_h, render_w = int(input_intrs[0, 0, 1, 2] * 2), int(input_intrs[0, 0, 0, 2] * 2)
        N_target = target_image.shape[1]

        query_points, _ = self.renderer.get_query_points(input_flame_params, device=input_image.device)
        assert query_points.shape[1] == N_input, "Query points should be [B, N_input, N_points, 3]"
        
        # Construct camera parameters dictionary for input frames
        camera_params = {
            'c2w': input_c2ws,
            'intrinsic': input_intrs
        }
        
        # Get features for all frames
        latent_points, intermediate_outputs = self.forward_latent_points(
            input_image,
            query_points=query_points,
            flame_params=input_flame_params,
            camera_params=camera_params,
        )

        # Track head forward pass
        if self.use_tracking:
            track_list, vis, track_conf = self.track_head(intermediate_outputs, images=input_image, patch_start_idx=self.patch_start_idx, query_points=landmarks[:, 0])
        else:
            track_list, vis, track_conf = None, None, None

        # Confidence head forward pass
        if self.use_confidence:
            _, conf = self.confidence_head(intermediate_outputs, images=input_image, patch_start_idx=self.patch_start_idx)
        else:
            conf = None
        
        # Clean up intermediate outputs immediately
        del intermediate_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use first frame for single frame processing
        B, N_input, N_points, C = latent_points.shape
        single_frame_idx = 0  # Always use first frame
        single_latent_points = latent_points[:, single_frame_idx]  # [B, N_points, C]
        single_query_points = query_points[:, single_frame_idx]  # [B, N_points, 3]

        assert len(single_latent_points.shape) == 3, "Latent points should be [B, N_points, C]"

        # Single Frame Latent Points 
        out = self._render_multiple_frames(
            latent_points=single_latent_points,
            query_points=single_query_points,
            inf_flame_params=inf_flame_params,
            c2ws=target_c2ws,
            intrs=target_intrs,
            bg_colors=target_bg_colors,
            render_h=render_h,
            render_w=render_w,
            N_inf=N_target,
            input_indices=[single_frame_idx]
        )

        # Single frame doesn't need recon_input for confidence loss
        out['single_recon_input_comp_rgb'] = None
        out['single_recon_input_comp_mask'] = None

        if self.use_multi_frame_pc:
            # Multi-frame point cloud processing - use first N frames
            num_sliced_frames = min(self.num_sliced_frames, N_input)
            sliced_frame_indices = list(range(num_sliced_frames))
            sliced_latent_points = latent_points[:, :num_sliced_frames]  # [B, num_sliced_frames, N_points, C]
            sliced_query_points = query_points[:, :num_sliced_frames]  # [B, num_sliced_frames, N_points, 3]
            
            # Reshape to concatenate frames along the points dimension
            B_sliced, num_frames, N_points, C = sliced_latent_points.shape
            sliced_latent_points = sliced_latent_points.reshape(B_sliced, num_frames * N_points, C)
            sliced_query_points = sliced_query_points.reshape(B_sliced, num_frames * N_points, 3)
            
            assert sliced_latent_points.shape[1] == num_sliced_frames * N_points, f"Sliced latent points should be [B, num_sliced_frames*N_points, C], but got {sliced_latent_points.shape}"
            assert sliced_query_points.shape[1] == num_sliced_frames * N_points, f"Sliced query points should be [B, num_sliced_frames*N_points, 3], but got {sliced_query_points.shape}"

            multi_pc_out = self._render_multiple_frames(
                latent_points=sliced_latent_points,
                query_points=sliced_query_points,
                inf_flame_params=inf_flame_params,
                c2ws=target_c2ws,
                intrs=target_intrs,
                bg_colors=target_bg_colors,
                render_h=render_h,
                render_w=render_w,
                N_inf=N_target,
                input_indices=sliced_frame_indices
            )

            # Only compute recon_input if confidence is enabled
            if self.use_confidence:
                # Determine how many frames to use for confidence loss
                conf_frames = min(self.conf_loss_frames, N_input)
                
                # Use first conf_frames for confidence loss calculation
                conf_frame_indices = list(range(conf_frames))
                
                # Use first conf_frames for confidence loss calculation
                conf_latent_points = latent_points[:, :conf_frames].reshape(B, -1, latent_points.shape[-1])
                conf_query_points = query_points[:, :conf_frames].reshape(B, -1, query_points.shape[-1])
                
                # Slice input parameters to match conf_frames
                conf_input_flame_params = {}
                for k, v in input_flame_params.items():
                    if isinstance(v, torch.Tensor):
                        conf_input_flame_params[k] = v[:, :conf_frames]
                    else:
                        conf_input_flame_params[k] = v
                
                recon_input_out = self._render_multiple_frames(
                    latent_points=conf_latent_points,
                    query_points=conf_query_points,
                    inf_flame_params=conf_input_flame_params,
                    c2ws=input_c2ws[:, conf_frame_indices],
                    intrs=input_intrs[:, conf_frame_indices],
                    bg_colors=input_bg_colors[:, conf_frame_indices],
                    render_h=render_h,
                    render_w=render_w,
                    N_inf=conf_frames,
                    input_indices=conf_frame_indices
                )

                out['sliced_comp_rgb'] = multi_pc_out['comp_rgb']
                out['sliced_comp_mask'] = multi_pc_out['comp_mask']
                out['recon_input_comp_rgb'] = recon_input_out['comp_rgb']
                out['recon_input_comp_mask'] = recon_input_out['comp_mask']
                out['conf_frame_indices'] = conf_frame_indices
                
                # Clean up multi-frame variables immediately
                del multi_pc_out, recon_input_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                out['sliced_comp_rgb'] = multi_pc_out['comp_rgb']
                out['sliced_comp_mask'] = multi_pc_out['comp_mask']
                out['recon_input_comp_rgb'] = None
                out['recon_input_comp_mask'] = None
                
                # Clean up multi-frame variables immediately
                del multi_pc_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Use single frame for all outputs when multi-frame is disabled
            out['sliced_comp_rgb'] = out['comp_rgb']
            out['sliced_comp_mask'] = out['comp_mask']
            
            # Only compute recon_input if confidence is enabled
            if self.use_confidence:
                # For recon_input, use single frame with input parameters
                recon_input_out = self._render_multiple_frames(
                    latent_points=single_latent_points,
                    query_points=single_query_points,
                    inf_flame_params=input_flame_params,
                    c2ws=input_c2ws,
                    intrs=input_intrs,
                    bg_colors=input_bg_colors,
                    render_h=render_h,
                    render_w=render_w,
                    N_inf=N_input,
                    input_indices=[single_frame_idx]
                )
                out['recon_input_comp_rgb'] = recon_input_out['comp_rgb']
                out['recon_input_comp_mask'] = recon_input_out['comp_mask']
                
                # Clean up single frame variables
                del recon_input_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                out['recon_input_comp_rgb'] = None
                out['recon_input_comp_mask'] = None

        # Add tracking outputs only if tracking is enabled
        if self.use_tracking and track_list is not None:
            out['track_list'] = track_list[-1]
            out['vis'] = vis
            out['track_conf'] = track_conf
        else:
            out['track_list'] = None
            out['vis'] = None
            out['track_conf'] = None
            
        # Add confidence output only if confidence is enabled
        if self.use_confidence:
            out['conf'] = conf
        else:
            out['conf'] = None
            
        out['num_input_frames'] = N_input
        
        # Clean up temporary variables
        del single_latent_points, single_query_points
        if self.use_multi_frame_pc:
            del sliced_latent_points, sliced_query_points
        del latent_points, query_points  # Clean up the original large tensors
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return out
    
    @torch.no_grad()
    def infer_images(self, image, input_c2ws, input_intrs, target_c2ws, target_intrs, target_bg_colors, input_flame_params, inf_flame_params, render_h=512, render_w=512):
        import time
        
        B, N_input = image.shape[:2]
        N_target = target_c2ws.shape[1]
        
        print(f"[INFER] Processing {N_input} input frames, {N_target} target views")
        print(f"[INFER] Render resolution: {render_h}x{render_w}")
        
        # Step 1: Get query points
        query_points, _ = self.renderer.get_query_points(input_flame_params, device=image.device)
        
        # Step 2: Forward latent points (most expensive step)
        latent_points_start = time.time()
        # Construct camera parameters dictionary for INPUT frames (for encoder)
        camera_params = {
            'c2w': input_c2ws,
            'intrinsic': input_intrs
        }
        latent_points, _ = self.forward_latent_points(image, query_points=query_points, flame_params=input_flame_params, camera_params=camera_params)
        latent_points_time = time.time() - latent_points_start
        print(f"[INFER] forward_latent_points: {latent_points_time:.3f}s")
        
        query_points = query_points.reshape(B, -1, query_points.shape[-1])
        latent_points = latent_points.reshape(B, -1, latent_points.shape[-1])

        render_start = time.time()
        
        params_start = time.time()
        
        # Debug: Print original flame params shapes
        print(f"[INFER] Debug - Original inf_flame_params shapes:")
        for k, v in inf_flame_params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
        
        print(f"[INFER] Debug - Original input_flame_params shapes:")
        for k, v in input_flame_params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Use inf_flame_params directly without modification (like in inference code)
        batch_flame_params = inf_flame_params.copy()
        print(f"[INFER] Using inf_flame_params directly")
        
        # Debug: Print final batch_flame_params shapes
        print(f"[INFER] Debug - Final batch_flame_params shapes:")
        for k, v in batch_flame_params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
        
        params_time = time.time() - params_start
        print(f"[INFER] flame_params_processing: {params_time:.3f}s")

        convert_start = time.time()
        latent_points = latent_points.float()
        query_points = query_points.float()
        batch_c2ws = target_c2ws.float() 
        batch_intrs = target_intrs.float()
        batch_bg_colors = target_bg_colors.float()
        batch_flame_params = {k: v.float() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch_flame_params.items()}
        convert_time = time.time() - convert_start
        print(f"[INFER] data_conversion: {convert_time:.3f}s")
        
        # Handle multi-frame point cloud concatenation
        if self.use_multi_frame_pc and N_input > 1:
            # Multi-frame mode: concatenate all frames' latent points
            # Check if latent_points is still 4D before reshaping
            if len(latent_points.shape) == 4:
                B_all, N_frames, N_points, C = latent_points.shape
                latent_points = latent_points.reshape(B_all, N_frames * N_points, C)
                query_points = query_points.reshape(B_all, N_frames * N_points, 3)
                print(f"[INFER] Multi-frame PC mode: concatenated {N_frames} frames, total points: {N_frames * N_points}")
            else:
                # Already reshaped, just print info
                print(f"[INFER] Multi-frame PC mode: already concatenated, shape: {latent_points.shape}")
        else:
            # Single-frame mode: use only the first frame
            if N_input > 1:
                if len(latent_points.shape) == 4:
                    latent_points = latent_points[:, 0]  # [B, N_points, C]
                    query_points = query_points[:, 0]    # [B, N_points, 3]
                print(f"[INFER] Single-frame PC mode: using first frame only")
            else:
                print(f"[INFER] Single input frame mode")

        gs_start = time.time()

        gs_model_list, curr_query_points, curr_flame_params, _ = self.renderer.forward_gs(
            gs_hidden_features=latent_points, 
            query_points=query_points, 
            flame_data=input_flame_params
        )
        gs_time = time.time() - gs_start
        print(f"[INFER] forward_gs: {gs_time:.3f}s")

        animate_start = time.time()
        
        render_res = self.renderer.forward_animate_gs(
            gs_model_list, 
            curr_query_points, 
            batch_flame_params, 
            batch_c2ws,
            batch_intrs, 
            render_h, 
            render_w, 
            batch_bg_colors,
            num_input_frames=N_input
        )
        animate_time = time.time() - animate_start
        print(f"[INFER] forward_animate_gs: {animate_time:.3f}s")
        
        total_render_time = time.time() - render_start
        print(f"[INFER] Total rendering time: {total_render_time:.3f}s")
        
        # Store timing information in output
        out = render_res
        out['total_render_time'] = total_render_time
        
        out = render_res
        if "comp_rgb" in out and isinstance(out["comp_rgb"], torch.Tensor):
            # Reshape if needed: [B, Nv, 3, H, W] -> [Nv, H, W, 3]
            if len(out["comp_rgb"].shape) == 5:
                out["comp_rgb"] = out["comp_rgb"][0].permute(0, 2, 3, 1)
        if "comp_mask" in out and isinstance(out["comp_mask"], torch.Tensor):
            if len(out["comp_mask"].shape) == 5:
                out["comp_mask"] = out["comp_mask"][0].permute(0, 2, 3, 1)
        if "comp_depth" in out and isinstance(out["comp_depth"], torch.Tensor):
            if len(out["comp_depth"].shape) == 5:
                out["comp_depth"] = out["comp_depth"][0].permute(0, 2, 3, 1)
        
        out['cano_gs_lst'] = gs_model_list
        
        return out