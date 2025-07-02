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
from safetensors.torch import load_file

from FastAvatar.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from FastAvatar.models.alternating_cross_attn import AlternatingCrossAttn
from diffusers.utils import is_torch_version
from FastAvatar.models.heads.dpt_head import DPTHead
from FastAvatar.models.track_head import TrackHead

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFastAvatar(nn.Module):
    """
    Model focusing on image encoding and points embedding,
    maintaining full compatibility with LAM pretrained weights.
    Now supports multi-frame processing.
    """
    def __init__(self,
                 transformer_dim: int = 1024, 
                 transformer_layers: int = 10,
                 transformer_heads: int = 16,
                 aa_order: list = ["frame", "global"],
                 aa_block_size: int = 1,
                 tf_grad_ckpt=True,
                 pretrained_model_path: str = None,
                 encoder_grad_ckpt=True,
                 encoder_freeze: bool = True, encoder_type: str = 'dinov2_fusion',
                 encoder_model_name: str = 'dinov2_vitl14_reg',
                 encoder_feat_dim: int = 1024,
                 num_pcl: int=2048, pcl_dim: int=1024,
                 human_model_path="./model_zoo/human_parametric_models",
                 flame_subdivide_num=1,
                 flame_type="flame",
                 gs_query_dim=1024,
                 gs_use_rgb=True,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=0.2,
                 gs_clip_scaling=0.01,
                 shape_param_dim=10,
                 expr_param_dim=10,
                 fix_opacity=False,
                 fix_rotation=False,
                 enable_lora: bool = True,
                 lora_cfg: dict = None,
                 use_flame_tokens: bool = True,
                 flame_encoder_config: dict = None,
                 use_multi_frame_pc: bool = True,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.use_multi_frame_pc = use_multi_frame_pc

        # frozen Parameters
        self.encoder_frozen = kwargs.get('encoder_frozen', False)
        self.pcl_frozen = kwargs.get('pcl_frozen', False)
        self.gs_frozen = kwargs.get('gs_frozen', False)

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
                                     shape_param_dim=shape_param_dim,
                                     expr_param_dim=expr_param_dim,
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
        if use_flame_tokens:
            self.patch_start_idx = 1
        else:
            self.patch_start_idx = 0

        # Load pretrained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"\nLoading pretrained model from {pretrained_model_path}")
            state_dict = load_file(pretrained_model_path)
            
            # Extract parameters for each component based on their paths
            encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            pcl_embed_dict = {k.replace('pcl_embed.', ''): v for k, v in state_dict.items() if k.startswith('pcl_embed.')}
            renderer_dict = {k.replace('renderer.', ''): v for k, v in state_dict.items() if k.startswith('renderer.')}
            pretrain_attn_dict = {k.replace('transformer.', ''): v for k, v in state_dict.items() if k.startswith('transformer.')}
            
            # Load parameters for each component
            self.encoder.load_state_dict(encoder_dict, strict=False)
            self.pcl_embed.load_state_dict(pcl_embed_dict, strict=False)
            self.renderer.load_state_dict(renderer_dict, strict=False)

            if self.encoder_frozen:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                logger.info("Encoder parameters have been frozen")

            # Freeze pcl_embed parameters
            if self.pcl_frozen:
                for param in self.pcl_embed.parameters():
                    param.requires_grad = False
                logger.info("Pcl_embed parameters have been frozen")

            # Freeze renderer parameters
            if self.gs_frozen:
                for param in self.renderer.parameters():
                    param.requires_grad = False
                logger.info("Renderer parameters have been frozen")
            
            # Check transformer layers
            pretrained_layers = len([k for k in pretrain_attn_dict.keys() if k.startswith('layers.')])
            logger.info(f"Pretrained transformer has {pretrained_layers} layers")
            
            if transformer_layers > pretrained_layers:
                raise ValueError(f"Requested {transformer_layers} transformer layers, but pretrained model only has {pretrained_layers} layers")
            
            # Create new state dict with only the requested number of layers
            frame_attn_dict = {k: v for k, v in pretrain_attn_dict.items() 
                             if not k.startswith('layers.') or 
                             int(k.split('.')[1]) < transformer_layers}
            
            # Initialize AlternatingAttn
            self.intermediate_layer_idx = kwargs.get("intermediate_layer_idx", [2,5,8])  # Empty list by default
            self.post_fusion = kwargs.get("post_fusion", False)
            self.transformer = AlternatingCrossAttn(
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                inner_dim=transformer_dim,
                cond_dim=transformer_dim,
                gradient_checkpointing=self.gradient_checkpointing,
                pretrained_attn_dict=frame_attn_dict,
                aa_order=aa_order,
                aa_block_size=aa_block_size,
                lora_cfg=lora_cfg,
                intermediate_layer_idx=self.intermediate_layer_idx,
                use_flame_tokens=use_flame_tokens,
                flame_encoder_config=flame_encoder_config,
                enable_lora=enable_lora,
                post_fusion=self.post_fusion
            )
        
        # Track Head
        self.vggt_model_path = kwargs.get("vggt_model_path", None)
        self.intermediate_layer_idx = kwargs.get("intermediate_layer_idx", [2,5,8])
        self.track_head = TrackHead(
            dim_in = transformer_dim * 2, 
            intermediate_layer_idx=self.intermediate_layer_idx,
            use_flame_tokens=use_flame_tokens
        )

        # confidence head
        self.confidence_head = DPTHead(dim_in=transformer_dim*2, output_dim=1, activation='inv_log', conf_activation='expp1')
        
        if self.vggt_model_path and os.path.exists(self.vggt_model_path):
            logger.info(f"\nLoading track_head weights from VGGT model: {self.vggt_model_path}")
            vggt_state_dict = load_file(self.vggt_model_path)
            track_head_dict = {k.replace('track_head.', ''): v for k, v in vggt_state_dict.items() if k.startswith('track_head.')}
            if track_head_dict:
                self.track_head.load_state_dict(track_head_dict, strict=False)
                logger.info("Successfully loaded track_head weights from VGGT model")
            else:
                logger.warning("No track_head weights found in VGGT model")

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
    
    def forward_transformer(self, image_feats, camera_embeddings, query_points, query_feats=None, flame_params=None):
        """
        Args:
            image_feats: [B, N_input, H*W, C]
            query_points: [B, N_points, 3]
            flame_params: Dictionary containing FLAME parameters for encoding into tokens
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
        
        x, cond, intermediate_outputs = self.transformer(x, image_feats, mod=camera_embeddings, flame_params=flame_params)
        
        return x, cond, intermediate_outputs

    @torch.compile
    def forward_latent_points(self, image, query_points=None, flame_params=None):
        """
        Process image and query points to get latent features
        Args:
            image: [B, N_input, C_img, H_img, W_img] - N_input: Output frame number
            query_points: [B, N_points, 3]
            flame_params: Dictionary containing FLAME parameters for encoding into tokens
        Returns:
            points_embedding: [B, N_inf, N_points, C]
            image_feats: [B, N_input, H*W, C]
            intermediate_outputs: List of intermediate layer outputs
        """
        B, N_input = image.shape[:2]
        B2, N_points = query_points.shape[:2]
        assert B == B2, "Batch size must match"
        assert N_input == N_points, "Frame number must match query points number"
        
        # encode image
        image_feats = self.forward_encode_image(image)
        
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # Convert query_points to the same dtype as image_feats
        query_points = query_points.to(image_feats.dtype)
        image_feats = image_feats.view(B, N_input, *image_feats.shape[1:])
        # Get transformer output
        latent_points, cond, intermediate_outputs = self.forward_transformer(image_feats, camera_embeddings=None, query_points=query_points, flame_params=flame_params)

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
                latent_points = latent_points.float()
                query_points = query_points.float()
                frame_c2ws = c2ws[:, f:f+1].float()
                frame_intrs = intrs[:, f:f+1].float()
                frame_bg_colors = bg_colors[:, f:f+1].float()
                frame_flame_params = {k: v.float() if isinstance(v, torch.Tensor) else v 
                                    for k, v in frame_flame_params.items()}

                # Render this frame
                render_res = self.renderer(
                    gs_hidden_features=latent_points,
                    query_points=query_points,
                    flame_data=frame_flame_params,
                    c2w=frame_c2ws,
                    intrinsic=frame_intrs,
                    height=render_h,
                    width=render_w,
                    background_color=frame_bg_colors,
                    num_input_frames=len(input_indices),
                )
                render_res_list.append(render_res)

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
        # query_points = query_points[:, :1].repeat(1, N_input, 1, 1) # [B, N_input, N_points, 3] anchor the first frame
        
        # Get features for all frames
        latent_points, intermediate_outputs = self.forward_latent_points(
            input_image,
            query_points=query_points,
            flame_params=input_flame_params,
        )

        track_list, vis, track_conf = self.track_head(intermediate_outputs, images=input_image, patch_start_idx=self.patch_start_idx, query_points=landmarks[:, 0])

        _, conf = self.confidence_head(intermediate_outputs, images=input_image, patch_start_idx=self.patch_start_idx)
        
        del intermediate_outputs # clear memory
        
        # Use uid to generate deterministic random frame index
        B, N_input, N_points, C = latent_points.shape
        sequence_num = int(uid[0][0].split('/')[-1])  # Get the last number in the path
        rng = np.random.RandomState(sequence_num)
        random_frame_idx = rng.randint(0, N_input)
        single_latent_points = latent_points[:, random_frame_idx]  # [B, N_points, C]
        single_query_points = query_points[:, random_frame_idx]  # [B, N_points, 3]

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
            input_indices=[random_frame_idx]
        )

        if self.use_multi_frame_pc:
            # Multi-frame point cloud processing
            num_sliced_frames = self.num_sliced_frames if self.num_sliced_frames < N_input else N_input
            sliced_frame_indices = rng.choice(N_input, size=num_sliced_frames, replace=False)
            sliced_latent_points = latent_points[:, sliced_frame_indices]  # [B, num_sliced_frames, N_points, C]
            sliced_query_points = query_points[:, sliced_frame_indices]  # [B, num_sliced_frames, N_points, 3]
            sliced_latent_points = sliced_latent_points.reshape(B, -1, sliced_latent_points.shape[-1])
            sliced_query_points = sliced_query_points.reshape(B, -1, sliced_query_points.shape[-1])
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

            recon_input_out = self._render_multiple_frames(
                latent_points=sliced_latent_points,
                query_points=sliced_query_points,
                inf_flame_params=input_flame_params,
                c2ws=input_c2ws,
                intrs=input_intrs,
                bg_colors=input_bg_colors,
                render_h=render_h,
                render_w=render_w,
                N_inf=N_input,
                input_indices=sliced_frame_indices
            )

            out['sliced_comp_rgb'] = multi_pc_out['comp_rgb']
            out['sliced_comp_mask'] = multi_pc_out['comp_mask']
            out['recon_input_comp_rgb'] = recon_input_out['comp_rgb']
            out['recon_input_comp_mask'] = recon_input_out['comp_mask']
        else:
            # Use single frame for all outputs when multi-frame is disabled
            out['sliced_comp_rgb'] = out['comp_rgb']
            out['sliced_comp_mask'] = out['comp_mask']
            
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
                input_indices=[random_frame_idx]
            )
            out['recon_input_comp_rgb'] = recon_input_out['comp_rgb']
            out['recon_input_comp_mask'] = recon_input_out['comp_mask']

        out['latent_points'] = latent_points
        out['query_points'] = query_points
        out['track_list'] = track_list[-1]
        out['vis'] = vis
        out['track_conf'] = track_conf
        out['conf'] = conf
        
        # Clean up temporary variables
        del single_latent_points, single_query_points
        if self.use_multi_frame_pc:
            del sliced_latent_points, sliced_query_points, multi_pc_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return out
    
    @torch.no_grad()
    def infer_iamges(self, image, c2ws, intrs, bg_colors, inf_flame_params):
        B, N_input = image.shape[:2]
        render_h, render_w = int(intrs[0, 0, 1, 2] * 2), int(intrs[0, 0, 0, 2] * 2)
        N_target = c2ws.shape[1]
        
        query_points, _ = self.renderer.get_query_points(inf_flame_params, device=image.device)
        query_points = query_points[:, :1].repeat(1, N_input, 1, 1) # [B, N_input, N_points, 3] anchor the first frame
        B2, _, N_points, _ = query_points.shape
        assert B == B2, "Batch size must match"
        latent_points, _ = self.forward_latent_points(image, query_points=query_points)

        query_points = query_points.view(B, -1, 3)
        latent_points = latent_points.view(B, -1, latent_points.shape[-1])
        assert query_points.shape[1] == N_input * N_points, "Query points should be [B, N_input*N_points, 3]"
        assert latent_points.shape[1] == N_input * N_points, "Latent points should be [B, N_input*N_points, C]"

        render_res_list = []
        for view_idx in range(N_target):

            # Get frame-specific parameters
            frame_flame_params = {}
            for k, v in inf_flame_params.items():
                if isinstance(v, torch.Tensor):
                    if k == "betas":
                        frame_flame_params[k] = v
                    else:
                        frame_flame_params[k] = v[:, view_idx:view_idx+1]
                else:
                    frame_flame_params[k] = v

            # Convert all inputs to float32 for rasterization
            latent_points = latent_points.float()
            query_points = query_points.float()
            frame_c2ws = c2ws[:, view_idx:view_idx+1].float()
            frame_intrs = intrs[:, view_idx:view_idx+1].float()
            frame_bg_colors = bg_colors[:, view_idx:view_idx+1].float()
            frame_flame_params = {k: v.float() if isinstance(v, torch.Tensor) else v 
                                for k, v in frame_flame_params.items()}
            
            gs_model_list, curr_query_points, curr_flame_params, _ = self.renderer.forward_gs(gs_hidden_features=latent_points, 
                                                    query_points=query_points, 
                                                    flame_data=frame_flame_params)

            render_res = self.renderer.forward_animate_gs(gs_model_list, 
                                                          curr_query_points, 
                                                          curr_flame_params, 
                                                          frame_c2ws,
                                                          frame_intrs, 
                                                          render_h, 
                                                          render_w, 
                                                          frame_bg_colors,
                                                          num_input_frames=N_input)
            render_res_list.append(render_res)
        
        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(0, 2, 3, 1)  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3] 
            else:
                out[k] = v
        out['cano_gs_lst'] = gs_model_list
        return out