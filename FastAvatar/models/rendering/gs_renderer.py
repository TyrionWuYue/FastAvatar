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
from collections import defaultdict
try:
    from diff_gaussian_rasterization_wda import GaussianRasterizationSettings, GaussianRasterizer
except:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from FastAvatar.models.rendering.flame_model.flame import FlameHeadSubdivided
from pytorch3d.transforms import matrix_to_quaternion
from FastAvatar.models.rendering.utils.typing import *
from FastAvatar.models.rendering.utils.utils import trunc_exp, MLP
from FastAvatar.models.rendering.gaussian_model import GaussianModel
from einops import rearrange, repeat
os.environ["PYOPENGL_PLATFORM"] = "egl"

inverse_sigmoid = lambda x: np.log(x / (1 - x))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y

 
class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height, width, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = int(height)
        self.width = int(width)
        self.world_view_transform = w2c.transpose(0, 1)
        self.intrinsic = intrinsic

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(w2c.device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width)


class GSLayer(nn.Module):
    def __init__(self, in_channels, use_rgb, 
                 clip_scaling=0.2, 
                 init_scaling=-5.0,
                 scale_sphere=False,
                 init_density=0.1,
                 sh_degree=None, 
                 xyz_offset=True,
                 restrict_offset=True,
                 xyz_offset_max_step=None,
                 fix_opacity=False,
                 fix_rotation=False,
                 use_fine_feat=False,
                 pred_res=False,
                 ):
        super().__init__()
        self.clip_scaling = clip_scaling
        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat
        self.scale_sphere = scale_sphere
        self.pred_res = pred_res
        
        self.attr_dict ={
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3 if not scale_sphere else 1,
            "xyz": 3,
            "opacity": None,
            "rotation": None 
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4
        
        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                if pred_res:
                    layer = nn.Linear(in_channels+out_ch, out_ch)
                else:
                    layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer
            
        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer
        

            
    def forward(self, x, pts, x_fine=None, gs_raw_attr=None, ret_raw=False, vtx_sym_idxs=None):
        assert len(x.shape) == 2
        ret = {}
        if ret_raw:
            raw_attr = {}
        ori_x = x
        for k in self.attr_dict:
            # if vtx_sym_idxs is not None and k in ["shs", "scaling", "opacity"]:
            if vtx_sym_idxs is not None and k in ["shs", "scaling", "opacity", "rotation"]:
                # print("==="*16*3, "\n\n\n"+"use sym mean.", "\n"+"==="*16*3)
                # x = (x + x[vtx_sym_idxs.to(x.device), :]) / 2.
                x = ori_x[vtx_sym_idxs.to(x.device), :]
            else:
                x = ori_x
            layer =self.out_layers[k]
            if self.pred_res and (not self.fix_opacity or k != "opacity") and (not self.fix_rotation or k != "rotation"):
                v = layer(torch.cat([gs_raw_attr[k], x], dim=-1))
                v = gs_raw_attr[k] + v
            else:
                v = layer(x)
            if ret_raw:
                raw_attr[k] = v 
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(torch.eye(3).type_as(x)[None,: , :].repeat(x.shape[0], 1, 1)) # constant rotation
                else:
                    # assert len(x.shape) == 2
                    v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.scale_sphere:
                    assert v.shape[-1] == 1
                    v = torch.cat([v, v, v], dim=-1)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = torch.sigmoid(v)
            elif k == "shs":
                if self.use_rgb:
                    v[..., :3] = torch.sigmoid(v[..., :3])
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # TODO check
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                ret["offset"] = v
                v = pts + v
            ret[k] = v
            
        if ret_raw:
            return GaussianModel(**ret), raw_attr
        else:
            return GaussianModel(**ret)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        embed = self.norm(embed)
        return embed   


class GS3DRenderer(nn.Module):
    def __init__(self, human_model_path, subdivide_num, smpl_type, feat_dim, query_dim, 
                 use_rgb, sh_degree, xyz_offset_max_step, mlp_network_config,
                 clip_scaling=0.2,
                 scale_sphere=False,
                 skip_decoder=False,
                 fix_opacity=False,
                 fix_rotation=False,
                 decode_with_extra_info=None,
                 gradient_checkpointing=False,
                 add_teeth=True,
                 teeth_bs_flag=False,
                 oral_mesh_flag=False,
                 **kwargs,
                 ):
        super().__init__()
        print(f"#########scale sphere:{scale_sphere}, add_teeth:{add_teeth}")
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.smpl_type = smpl_type
        assert self.smpl_type == "flame"
        self.sym_rend2 = True
        self.teeth_bs_flag = teeth_bs_flag
        self.oral_mesh_flag = oral_mesh_flag
        self.render_rgb = kwargs.get("render_rgb", True)
        print("==="*16*3, "\n Render rgb:", self.render_rgb, "\n"+"==="*16*3)
        
        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree
        if use_rgb:
            self.sh_degree = 0

        use_rgb = use_rgb

        self.flame_model = FlameHeadSubdivided(
            300,
            100,
            add_teeth=add_teeth,
            add_shoulder=False,
            flame_model_path=f'{human_model_path}/flame_assets/flame/flame2023.pkl',
            flame_lmk_embedding_path=f"{human_model_path}/flame_assets/flame/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{human_model_path}/flame_assets/flame/head_template_mesh.obj",
            flame_parts_path=f"{human_model_path}/flame_assets/flame/FLAME_masks.pkl",
            subdivide_num=subdivide_num,
            teeth_bs_flag=teeth_bs_flag,
            oral_mesh_flag=oral_mesh_flag
        )

        self.mlp_network_config = mlp_network_config
        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config)

        init_scaling = -5.0
        
        
        num_points = self.flame_model.vertex_num_upsampled
        self.gs_net = GSLayer(in_channels=query_dim,
                              use_rgb=use_rgb,
                              sh_degree=self.sh_degree,
                              clip_scaling=clip_scaling,
                              scale_sphere=scale_sphere,
                              init_scaling=init_scaling,
                              init_density=0.1,
                              xyz_offset=True,
                              restrict_offset=True,
                              xyz_offset_max_step=xyz_offset_max_step,
                              fix_opacity=fix_opacity,
                              fix_rotation=fix_rotation,
                              use_fine_feat=True if decode_with_extra_info is not None and decode_with_extra_info["type"] is not None else False,
                              )
        
    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        GSRSettings = GaussianRasterizationSettings
        GSR = GaussianRasterizer

        raster_settings = GSRSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GSR(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
        else:
            shs = gs.shs

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            raster_ret = rasterizer(
                means3D = means3D.float(),
                means2D = means2D.float(),
                shs = shs.float() if shs is not None else None,
                colors_precomp = colors_precomp.float() if colors_precomp is not None else None,
                opacities = opacity.float(),
                scales = scales.float(),
                rotations = rotations.float(),
                cov3D_precomp = cov3D_precomp
            )
        rendered_image, radii, rendered_depth, rendered_alpha = raster_ret

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            'comp_mask': rendered_alpha.permute(1, 2, 0),
            'comp_depth': rendered_depth.permute(1, 2, 0),
        }

        return ret
            
    def animate_gs_model(self, gs_attr: GaussianModel, query_points, flame_data, num_input_frames, debug=False):
        """
        query_points: [N, 3]
        """
        device = gs_attr.xyz.device
        if debug:
            N = gs_attr.xyz.shape[0]
            gs_attr.xyz = torch.ones_like(gs_attr.xyz) * 0.0
            
            rotation = matrix_to_quaternion(torch.eye(3).float()[None, :, :].repeat(N, 1, 1)).to(device) # constant rotation
            opacity = torch.ones((N, 1)).float().to(device) # constant opacity

            gs_attr.opacity = opacity
            gs_attr.rotation = rotation
            # gs_attr.scaling = torch.ones_like(gs_attr.scaling) * 0.05
            # print(gs_attr.shs.shape)

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            # mean_3d = query_points + gs_attr.xyz  # [N, 3]
            mean_3d = gs_attr.xyz  # [N, 3]
            
            num_view = flame_data["expr"].shape[0]  # [Nv, 100]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)

            if self.teeth_bs_flag:
                expr = torch.cat([flame_data['expr'], flame_data['teeth_bs']], dim=-1)
            else:
                expr = flame_data["expr"]
            ret = self.flame_model.animation_forward(v_cano=mean_3d,
                                                shape=flame_data["betas"].repeat(num_view, 1),
                                                expr=expr,
                                                rotation=flame_data["rotation"],
                                                neck=flame_data["neck_pose"],
                                                jaw=flame_data["jaw_pose"],
                                                eyes=flame_data["eyes_pose"],
                                                translation=flame_data["translation"],
                                                zero_centered_at_root_node=False,
                                                return_landmarks=False,
                                                return_verts_cano=False,
                                                # static_offset=flame_data['static_offset'].to('cuda'),
                                                static_offset=None,
                                                num_input_frames=num_input_frames,
                                                )
            mean_3d = ret["animated"]
            
        gs_attr_list = []                                                                  
        for i in range(num_view):
            gs_attr_copy = GaussianModel(xyz=mean_3d[i],
                                    opacity=gs_attr.opacity, 
                                    rotation=gs_attr.rotation, 
                                    scaling=gs_attr.scaling,
                                    shs=gs_attr.shs,
                                    offset=gs_attr.offset) # [N, 3]
            gs_attr_list.append(gs_attr_copy)
        
        return gs_attr_list
        
    
    def forward_gs_attr(self, x, query_points, debug=False, x_fine=None, vtx_sym_idxs=None):
        """
        x: [N, C] Float[Tensor, "Np Cp"],
        query_points: [N, 3] Float[Tensor, "Np 3"]        
        """
        device = x.device
        if self.mlp_network_config is not None:
            x = self.mlp_net(x)
            if x_fine is not None:
                x_fine = self.mlp_net(x_fine)
        gs_attr: GaussianModel = self.gs_net(x, query_points, x_fine, vtx_sym_idxs=vtx_sym_idxs)
        return gs_attr
            

    def get_query_points(self, flame_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                # print(flame_data["betas"].shape, flame_data["face_offset"].shape, flame_data["joint_offset"].shape)
                # positions, _, transform_mat_neutral_pose = self.flame_model.get_query_points(flame_data, device=device)  # [B, N, 3]
                positions = self.flame_model.get_cano_verts(shape_params=flame_data["betas"])  # [B, N, 3]
                # print(f"positions shape:{positions.shape}")
                
        return positions, flame_data
    
    def query_latent_feat(self,
                          positions: Float[Tensor, "*B N1 3"],
                          flame_data,
                          latent_feat: Float[Tensor, "*B N2 C"],
                          extra_info):
        device = latent_feat.device
        gs_feats = latent_feat
        assert positions is not None
        return gs_feats, positions, flame_data

    def forward_single_batch(
        self,
        gs_list: list[GaussianModel],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "Nv 3"]],
        debug: bool=False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device
        for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
            out_list.append(self.forward_single_view(
                                gs_list[v_idx], 
                                Camera.from_c2w(c2w, intrinsic, height, width),
                                background_color[v_idx], 
                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list

        return out

    def get_sing_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[bidx]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
            if (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[bidx:bidx+1]  # e.g. betas: [B, 100] -> [1, 100]
        return smpl_data_single_batch
    
    def get_single_view_smpl_data(self, smpl_data, vidx):
        smpl_data_single_view = {}        
        for k, v in smpl_data.items():
            assert v.shape[0] == 1
            if k == "betas" or (k == "joint_offset") or (k == "face_offset") or (k == "transform_mat_neutral_pose"):
                smpl_data_single_view[k] = v  # e.g. betas: [1, 100] -> [1, 100]
            else:
                smpl_data_single_view[k] = v[:, vidx: vidx + 1]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
        return smpl_data_single_view
            
    def forward_gs(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        **kwargs):
                
        batch_size = gs_hidden_features.shape[0]
        
        query_gs_features, query_points, flame_data = self.query_latent_feat(query_points, flame_data, gs_hidden_features,
                                                                             additional_features)

        gs_model_list = []
        all_query_points = []
        for b in range(batch_size):
            all_query_points.append(query_points[b:b+1, :])
            if isinstance(query_gs_features, dict):
                ret_gs = self.forward_gs_attr(query_gs_features["coarse"][b], query_points[b], debug, 
                                                x_fine=query_gs_features["fine"][b], vtx_sym_idxs=None)
            else:
                ret_gs = self.forward_gs_attr(query_gs_features[b], query_points[b], debug, vtx_sym_idxs=None)

            gs_model_list.append(ret_gs)

        query_points = torch.cat(all_query_points, dim=0)

        return gs_model_list, query_points, flame_data, query_gs_features

    def forward_res_refine_gs(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        gs_raw_attr_list: list = None,
        **kwargs):
                
        batch_size = gs_hidden_features.shape[0]
        
        query_gs_features, query_points, flame_data = self.query_latent_feat(query_points, flame_data, gs_hidden_features,
                                                                             additional_features)

        gs_model_list = []
        for b in range(batch_size):
            gs_model = self.gs_refine_net(query_gs_features[b], query_points[b], x_fine=None, gs_raw_attr=gs_raw_attr_list[b])
            gs_model_list.append(gs_model)
        return gs_model_list, query_points, flame_data, query_gs_features

    def forward_animate_gs(self, gs_model_list, query_points, flame_data, c2w, intrinsic, height, width,
                           background_color, num_input_frames, debug=False):
        batch_size = len(gs_model_list)
        out_list = []

        for b in range(batch_size):
            gs_model = gs_model_list[b]
            query_pt = query_points[b]
            animatable_gs_model_list: list[GaussianModel] = self.animate_gs_model(gs_model,
                                                                                  query_pt,
                                                                                  self.get_sing_batch_smpl_data(flame_data, b),
                                                                                  num_input_frames=num_input_frames,
                                                                                  debug=debug)
            assert len(animatable_gs_model_list) == c2w.shape[1]
            out_list.append(self.forward_single_batch(
                animatable_gs_model_list,
                c2w[b],
                intrinsic[b],
                height, width,
                background_color[b] if background_color is not None else None, 
                debug=debug))
            
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v
                
        render_keys = ["comp_rgb", "comp_mask", "comp_depth"]
        for key in render_keys:
            out[key] = rearrange(out[key], "b v h w c -> b v c h w")
        
        return out

    def project_single_view_feats(self, img_vtx_ids, feats, nv, inter_feat=True):
        b, h, w, k = img_vtx_ids.shape
        c, ih, iw = feats.shape
        vtx_ids = img_vtx_ids
        if h != ih or w != iw:
            if inter_feat:
                feats = torch.nn.functional.interpolate(
                    rearrange(feats, "(b c) h w -> b c h w", b=1).float(), (h, w)
                ).squeeze(0)
                vtx_ids = rearrange(vtx_ids, "b (c h) w k -> (b k) c h w", c=1).long().squeeze(1)
            else:
                vtx_ids = torch.nn.functional.interpolate(
                    rearrange(vtx_ids, "b (c h) w k -> (b k) c h w", c=1).float(), (ih, iw), mode="nearest"
                ).long().squeeze(1)
        else:
            vtx_ids = rearrange(vtx_ids, "b h w k -> (b k) h w", b=1).long()
        vis_mask = vtx_ids > 0
        vtx_ids = vtx_ids[vis_mask]  # n
        vtx_ids = repeat(vtx_ids, "n -> n c", c=c)

        feats = repeat(feats, "c h w -> k h w c", k=k).to(vtx_ids.device)
        feats = feats[vis_mask, :] # n, c

        sums = torch.zeros((nv, c), dtype=feats.dtype, device=feats.device)
        counts = torch.zeros((nv), dtype=torch.int64, device=feats.device)

        sums.scatter_add_(0, vtx_ids, feats)
        one_hot = torch.ones_like(vtx_ids[:, 0], dtype=torch.int64).to(feats.device)
        counts.scatter_add_(0, vtx_ids[:, 0], one_hot)
        clamp_counts = counts.clamp(min=1)
        mean_feats = sums / clamp_counts.view(-1, 1) 
        return mean_feats
    
    def forward(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np 3"],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        num_input_frames: int = 1,
        **kwargs):
        
        # need shape_params of flame_data to get querty points and get "transform_mat_neutral_pose"
        gs_model_list, query_points, flame_data, query_gs_features = self.forward_gs(gs_hidden_features, query_points, flame_data=flame_data,
                                                                      additional_features=additional_features, debug=debug)

        out = self.forward_animate_gs(gs_model_list, query_points, flame_data, c2w, intrinsic, height, width, background_color, num_input_frames, debug)
        
        return out