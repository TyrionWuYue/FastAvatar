import os
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from safetensors.torch import load_file

from FastAvatar.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from FastAvatar.models.alternating_cross_attn import AlternatingCrossAttn
from FastAvatar.models.framepack_utils import FramePackCompressor
from FastAvatar.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
from diffusers.utils import is_torch_version

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFastAvatar(nn.Module):
    def __init__(self,
                 transformer_dim: int = 1024,
                 transformer_layers: int = 10,
                 transformer_heads: int = 16,
                 aa_order: list = ["global", "frame"],
                 tf_grad_ckpt=True,
                 pretrained_model_path: str = None,
                 encoder_path: str = None,
                 encoder_grad_ckpt=True,
                 encoder_freeze: bool = True,
                 source_image_res: int = 512,
                 encoder_model_name: str = 'dinov2_vitl14_reg',
                 encoder_feat_dim: int = 1024,
                 pcl_dim: int=1024,
                 human_model_path="./model_zoo/human_parametric_models",
                 renderer_freeze: bool = True,
                 flame_subdivide_num=1,
                 gs_query_dim=1024,
                 gs_use_rgb=True,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=0.2,
                 gs_clip_scaling=0.01,
                 fix_opacity=False,
                 fix_rotation=False,
                 gs_fusion: bool = True,
                 num_base_frames: int = 16,
                 if_framepack: bool = False,
                 framepack_compression_level: int = 4,
                 vggt_path: str = None,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.gs_fusion = gs_fusion
        self.rendering_chunk_size_train = kwargs.get("rendering_chunk_size_train", 16)
        self.rendering_chunk_size_infer = kwargs.get("rendering_chunk_size_infer", 128)
        self.num_base_frames = num_base_frames
        self.if_framepack = if_framepack
        self.framepack_compression_level = framepack_compression_level
        self.source_image_res = source_image_res

        # FramePack compressor (only used when if_framepack=True)
        if if_framepack:
            self.framepack_compressor = FramePackCompressor(
                in_channels=encoder_feat_dim,
                inner_dim=encoder_feat_dim,
                compression_level=framepack_compression_level
            )
        else:
            self.framepack_compressor = None

        # image encoder
        self.encoder = Dinov2FusionWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=self.encoder_feat_dim,
        )

        # learnable points embedding
        self.pcl_embed = PointEmbed(dim=pcl_dim)

        # Alternating cross Attention
        self.transformer = AlternatingCrossAttn(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=transformer_dim,
            gradient_checkpointing=self.gradient_checkpointing,
            aa_order=aa_order,
            patch_start_idx=0,
            if_framepack=if_framepack,
        )

        # renderer
        self.renderer = GS3DRenderer(human_model_path=human_model_path,
                                     subdivide_num=flame_subdivide_num,
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
                                     add_teeth=kwargs.get("add_teeth", True),
                                     teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
                                     oral_mesh_flag=kwargs.get("oral_mesh_flag", True),
                                     use_mesh_shading=kwargs.get('use_mesh_shading', False),
                                     render_rgb=kwargs.get("render_rgb", True),
                                     gs_pruning=kwargs.get("gs_pruning", False),
                                     )

        # Load pretrained model if available
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"Loading pretrained model from {pretrained_model_path}")
            state_dict = load_file(pretrained_model_path)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info(f"Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                logger.info(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
            logger.info(f"Pretrained model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        elif encoder_path and os.path.exists(encoder_path):
            # Training from scratch: load DINOv2 encoder weights
            logger.info(f"Training from scratch: loading encoder weights from {encoder_path}")
            encoder_state_dict = load_file(encoder_path) if encoder_path.endswith('.safetensors') else torch.load(encoder_path, map_location='cpu')
            encoder_dict = {k.replace('encoder.model.', '').replace('model.', ''): v for k, v in encoder_state_dict.items() if 'fusion_head' not in k}
            self.encoder.model.load_state_dict(encoder_dict, strict=False)
            logger.info("Encoder weights loaded for training from scratch")

        # Set parameter requires_grad
        for name, param in self.renderer.named_parameters():
            if name.startswith('flame_model.'):
                param.requires_grad = False
            elif name.startswith('mlp_net.') or name.startswith('gs_net.'):
                param.requires_grad = not renderer_freeze
            else:
                param.requires_grad = False
        

    def forward_encode_image(self, image):
        """
        Encode image features, supporting both single and multi-frame inputs
        Args:
            image: [B, N_frames, C_img, H_img, W_img]
        Returns:
            image_feats: [B, N_output, H*W, C] - Base frames features
            compressed_cond: [B, 1, compressed_tokens, C] or None - Compressed frame features (if framepack enabled)
            base_indices: list or None - Indices of base frames in original input (if framepack enabled)
            spatial_compression: int or None - Spatial compression ratio used (if framepack enabled)
        """
        B, N_frames, C_img, H_img, W_img = image.shape
        image = image.view(B * N_frames, C_img, H_img, W_img)

        tgt_size = (self.source_image_res // 14) * 14
        if H_img != tgt_size or W_img != tgt_size:
            image = F.resize(image, (tgt_size, tgt_size), antialias=True)
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
        
        # image_feats: [B*N_frames, H*W, C]
        _, HW, C = image_feats.shape
        
        # Unified FramePack processing: Always provide compression for all frame counts
        if self.if_framepack:
            image_feats = image_feats.view(B, N_frames, HW, C)

            # Always use first min(N_frames, num_base_frames) as base frames
            actual_base_frames = min(N_frames, self.num_base_frames)
            base_image_feats = image_feats[:, :actual_base_frames]
            base_indices = list(range(actual_base_frames))

            # Prepare 3D features for compression
            H_feat = int(HW ** 0.5)
            image_feats_3d = image_feats.view(B, N_frames, H_feat, H_feat, C)

            if N_frames > self.num_base_frames:
                compressed_input_3d = image_feats_3d[:, actual_base_frames:]
            else:
                compressed_input_3d = image_feats_3d

            compressed_features, spatial_compression = self.framepack_compressor(compressed_input_3d)

            compressed_cond = compressed_features.reshape(B, 1, -1, C)

            return base_image_feats, compressed_cond, base_indices, spatial_compression
        else:
            # No FramePack: simply slice all inputs to num_base_frames
            image_feats = image_feats.view(B, N_frames, HW, C)
            image_feats = image_feats[:, :self.num_base_frames]

            return image_feats, None, None, None

    def forward_transformer(self, image_feats, query_points, query_feats=None, compressed_cond=None, spatial_compression=None):
        """
        Args:
            image_feats: [B, N_input, H*W, C]
            query_points: [B, N_input, N_points, 3]
            query_feats: Optional query features
            compressed_cond: [B, 1, compressed_tokens, C] or None - Compressed frame features (if framepack enabled)
            spatial_compression: int or None - Spatial compression ratio used (if framepack enabled)
        Returns:
            latent_points: [B, N_input, N_points, C]
        """
        B, N_input = image_feats.shape[:2]

        # Reshape query_points for pcl_embed
        x = self.pcl_embed(query_points.reshape(B, -1, 3))
        x = x.reshape(B, query_points.shape[1], query_points.shape[2], x.shape[-1])
        if query_feats is not None:
            x = x + query_feats.to(image_feats.dtype)

        # Prepare compressed frame query points if exists
        compressed_x = x[:, N_input:N_input+1] if compressed_cond is not None else None
        if compressed_x is not None:
            x = x[:, :N_input]

        return self.transformer(x, image_feats, compressed_x=compressed_x, compressed_cond=compressed_cond, spatial_compression=spatial_compression)

    @torch.compile
    def forward_latent_points(self, image, input_flame_params):

        B, N_input = image.shape[:2]
        base_frames = min(self.num_base_frames, N_input)

        # Encode ALL frames + FramePack (compress non-base frames when if_framepack=True)
        base_feats, compressed_cond, _, spatial_compression = self.forward_encode_image(image)

        flame_for_query = input_flame_params.copy()
        if 'betas' not in flame_for_query and 'shape' in flame_for_query:
            flame_for_query['betas'] = flame_for_query['shape']
        query_points, _ = self.renderer.get_query_points(flame_for_query, device=image.device)

        # Prepare query points for transformer (base_frames + 1 compressed if exists)
        query_points_transformer = query_points[:, 0:1].repeat(1, base_frames, 1, 1)
        if compressed_cond is not None:
            query_points_transformer = torch.cat([query_points_transformer, query_points_transformer[:, 0:1]], dim=1)

        # Reconstruction Transformer
        latent_points = self.forward_transformer(
            base_feats,
            query_points_transformer,
            compressed_cond=compressed_cond,
            spatial_compression=spatial_compression
        )
        
        return latent_points, query_points_transformer

    def _render_multiple_frames(self, latent_points, query_points, inf_flame_params, c2ws, intrs, bg_colors, render_h, render_w, N_inf, chunk_size=16, input_indices=[0]):
        """
        Render multiple frames using the same latent points and query points with chunked rendering.

        Args:
            latent_points: [B, N_points, C] - Latent features for 3DGS
            query_points: [B, N_points, 3] - 3D query points
            inf_flame_params: Dict containing FLAME parameters for all frames
            c2ws: [B, N_inf, 4, 4] - Camera to world transformations
            intrs: [B, N_inf, 4, 4] - Camera intrinsics
            bg_colors: [B, N_inf, 3] - Background colors
            render_h, render_w: int - Render resolution
            N_inf: int - Number of frames to render
            chunk_size: int - Number of frames to render per chunk. If >= N_inf, render all frames at once

            rotation: [B, N_inf, 3] - Rotation parameters
            translation: [B, N_inf, 3] - Translation parameters
            expr: [B, N_inf, 100] - Expression parameters
            neck_pose: [B, N_inf, 3] - Neck pose parameters
            jaw_pose: [B, N_inf, 3] - Jaw pose parameters
            eyes_pose: [B, N_inf, 3] - Eyes pose parameters

        Returns:
            Dict containing concatenated render results for all frames
        """
        # Calculate number of chunks
        if chunk_size >= N_inf:
            # Render all frames at once
            chunk_size = N_inf
        num_chunks = (N_inf + chunk_size - 1) // chunk_size

        render_res_list = []
        try:
            for chunk_idx in range(num_chunks):
                # Calculate frame indices for this chunk
                start_frame = chunk_idx * chunk_size
                end_frame = min((chunk_idx + 1) * chunk_size, N_inf)

                # Extract chunk-specific parameters
                chunk_flame_params = {}
                for k, v in inf_flame_params.items():
                    if isinstance(v, torch.Tensor):
                        if k == "betas":
                            chunk_flame_params[k] = v[:, 0:1]  # [B, 1, ...]
                        else:
                            chunk_flame_params[k] = v[:, start_frame:end_frame]
                    else:
                        chunk_flame_params[k] = v

                # Convert all inputs to float32 for rasterization
                latent_points_f32 = latent_points.float()
                query_points_f32 = query_points.float()
                chunk_c2ws = c2ws[:, start_frame:end_frame].float()
                chunk_intrs = intrs[:, start_frame:end_frame].float()
                chunk_bg_colors = bg_colors[:, start_frame:end_frame].float()
                chunk_flame_params = {k: v.float() if isinstance(v, torch.Tensor) else v
                                    for k, v in chunk_flame_params.items()}

                # Render this chunk
                render_res = self.renderer(
                    gs_hidden_features=latent_points_f32,
                    query_points=query_points_f32,
                    flame_data=chunk_flame_params,
                    c2w=chunk_c2ws,
                    intrinsic=chunk_intrs,
                    height=render_h,
                    width=render_w,
                    background_color=chunk_bg_colors,
                    num_input_frames=len(input_indices),
                )
                render_res_list.append(render_res)

                # Clean up chunk-specific variables
                del render_res, chunk_flame_params, latent_points_f32, query_points_f32, chunk_c2ws, chunk_intrs, chunk_bg_colors

                # Force memory cleanup after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine results from all chunks
            out = {}  # Changed from defaultdict to regular dict
            for res in render_res_list:
                for k, v in res.items():
                    if k not in out:
                        out[k] = []
                    out[k].append(v)

            # Process each key in the output dictionary
            for k, v in out.items():
                if isinstance(v[0], torch.Tensor):
                    if k == "pruning_masks":
                        out[k] = torch.concat(v, dim=0)
                    else:  # Multi-dimensional tensors
                        out[k] = torch.concat(v, dim=1)
                        if k == "comp_rgb":
                            out[k] = out[k].permute(0, 1, 2, 3, 4) # [B, N_inf, H, W, C]
                    # Clean up the list immediately after concat to free memory
                    del v
                elif k == "gs_stats":
                    out[k] = v[0] if v else None
                else:
                    out[k] = v

            # Clean up render_res_list before returning
            del render_res_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return out
        finally:
            # Additional cleanup if needed (render_res_list already cleaned in the loop)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, input_image, target_image, input_c2ws, target_c2ws, input_intrs, target_intrs, input_bg_colors, target_bg_colors, landmarks, input_flame_params, inf_flame_params, uid):
        B, N_input = input_image.shape[:2]
        N_target = target_image.shape[1]
        
        # Obtain rendering resolution from target image to guarantee match for losses
        render_h, render_w = target_image.shape[-2:]
        
        # Forward: encoder + transformer, using GT FLAME params for query points
        latent_points, query_points = self.forward_latent_points(input_image, input_flame_params)
        
        del input_image, target_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Ground Truth Camera scaling (e.g. from native to render_w)
        gt_native_w = target_intrs[..., 0, 2:3] * 2.0
        gt_native_h = target_intrs[..., 1, 2:3] * 2.0
        gt_scale_w = render_w / gt_native_w.clamp(min=1.0)
        gt_scale_h = render_h / gt_native_h.clamp(min=1.0)
        target_intrs_scaled = target_intrs.clone()
        target_intrs_scaled[..., 0, 0] *= gt_scale_w.squeeze(-1)
        target_intrs_scaled[..., 1, 1] *= gt_scale_h.squeeze(-1)
        target_intrs_scaled[..., 0, 2] *= gt_scale_w.squeeze(-1)
        target_intrs_scaled[..., 1, 2] *= gt_scale_h.squeeze(-1)

        # Single path: GT Camera + GT FLAME (from FLAME Tracking)
        latent_flat = latent_points.reshape(B, -1, latent_points.shape[-1])
        query_flat = query_points.detach().reshape(B, -1, 3)
        render_kwargs = dict(
            render_h=render_h, render_w=render_w,
            N_inf=N_target,
            chunk_size=self.rendering_chunk_size_train,
            input_indices=list(range(latent_points.shape[1]))
        )

        out = self._render_multiple_frames(
            latent_points=latent_flat,
            query_points=query_flat,
            inf_flame_params=inf_flame_params,
            c2ws=target_c2ws,
            intrs=target_intrs_scaled,
            bg_colors=target_bg_colors,
            **render_kwargs
        )
        
        del input_c2ws, target_c2ws, input_intrs, target_intrs, landmarks, input_flame_params, latent_flat, query_flat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return out
    
    @torch.no_grad()
    def infer_images(self, image, input_c2ws, input_intrs, target_c2ws, target_intrs, target_bg_colors, input_flame_params, inf_flame_params=None, render_h=512, render_w=512):
        B, N_input = image.shape[:2]
        N_target = target_c2ws.shape[1]
        
        modeling_time = time.time()
        
        latent_points, query_points = self.forward_latent_points(image, input_flame_params)

        # Clean up input tensors immediately after forward_latent_points
        del image, input_c2ws, input_intrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        modeling_time = time.time() - modeling_time

        # Use all input frames (no slicing)
        num_input_frames = latent_points.shape[1]

        # Reshape to concatenate frames along the points dimension (same as forward)
        B, num_frames, N_points, C = latent_points.shape
        latent_points_reshaped = latent_points.reshape(B, num_frames * N_points, C)
        query_points_reshaped = query_points.reshape(B, num_frames * N_points, 3)

        # Rendering
        render_start = time.time()
        out = self._render_multiple_frames(
            latent_points=latent_points_reshaped,
            query_points=query_points_reshaped,
            inf_flame_params=inf_flame_params,
            c2ws=target_c2ws,
            intrs=target_intrs,
            bg_colors=target_bg_colors,
            render_h=render_h,
            render_w=render_w,
            N_inf=N_target,
            chunk_size=self.rendering_chunk_size_infer,  # Use inference chunk size
            input_indices=list(range(num_input_frames))
        )
        render_time = time.time() - render_start

        del latent_points_reshaped, query_points_reshaped, latent_points
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Store timing information in output
        out['modeling_time'] = modeling_time
        out['render_time'] = render_time
        
        # Reshape comp_rgb: [B, N_target, H, W, C] -> [N_target, H, W, 3]
        if "comp_rgb" in out and isinstance(out["comp_rgb"], torch.Tensor):
            if len(out["comp_rgb"].shape) == 5:
                out["comp_rgb"] = out["comp_rgb"][0].permute(0, 2, 3, 1)
        
        return out