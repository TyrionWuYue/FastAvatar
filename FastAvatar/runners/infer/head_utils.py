from collections import defaultdict
import os
import json
import numpy as np
from PIL import Image
import cv2
import torch


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y  
    return intrs    
    
def calc_new_tgt_size(cur_hw, tgt_size, multiply):
    ratio = tgt_size / min(cur_hw)
    tgt_size = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
    tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x

def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_size = tgt_size * aspect_standard, tgt_size
    tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x
    

def img_center_padding(img_np, pad_ratio):
    
    ori_w, ori_h = img_np.shape[:2]
    
    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)
    
    if len(img_np.shape) > 2:
        img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
    else:
        img_pad_np = np.zeros((w, h), dtype=np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[offset_h: offset_h + img_np.shape[0]:, offset_w: offset_w + img_np.shape[1]] = img_np
    
    return img_pad_np


def resize_image_keepaspect_np(img, max_tgt_size):
    """
    similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio  
    """
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop_according_to_mask(img, mask, aspect_standard, enlarge_ratio):
    """ 
        img: [H, W, 3]
        mask: [H, W]
    """ 
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("empty mask")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    
    center_x, center_y = img.shape[1]//2, img.shape[0]//2
    
    half_w = max(abs(center_x - x_min), abs(center_x -  x_max))
    half_h = max(abs(center_y - y_min), abs(center_y -  y_max))
    half_w_raw = half_w
    half_h_raw = half_h
    aspect = half_h / half_w

    if aspect >= aspect_standard:                
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    if half_h > center_y:
        half_w = round(half_h_raw / aspect_standard)
        half_h = half_h_raw
    if half_w > center_x:
        half_h = round(half_w_raw * aspect_standard)
        half_w = half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) >  0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
        half_h, half_w = round(enlarge_ratio_cur * half_h), round(enlarge_ratio_cur * half_w)

    assert half_h <= center_y
    assert half_w <= center_x
    assert abs(half_h / half_w - aspect_standard) < 0.03
    
    offset_x = center_x - half_w
    offset_y = center_y - half_h
    
    new_img = img[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
    new_mask = mask[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
    
    return  new_img, new_mask, offset_x, offset_y  
    

def preprocess_image(rgb_path, mask_path, intr, pad_ratio, bg_color, 
                            max_tgt_size, aspect_standard, enlarge_ratio,
                            render_tgt_size, multiply, need_mask=True,
                            get_shape_param=False):
    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()
    if pad_ratio > 0:
        rgb = img_center_padding(rgb, pad_ratio)

    rgb = rgb / 255.0
    if need_mask:
        if rgb.shape[2] < 4:
            if mask_path is not None:
                # mask = np.array(Image.open(mask_path))
                mask = (np.array(Image.open(mask_path)) > 180) * 255
            else:
                from rembg import remove
                mask = remove(rgb_raw[:, :, (2, 1, 0)])[:, :, -1]  # np require [bgr]
                print("rmbg mask: ", mask.min(), mask.max(), mask.shape)
            if pad_ratio > 0:
                mask = img_center_padding(mask, pad_ratio)
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
    else:
        # just placeholder
        mask = np.ones_like(rgb[:, :, 0])
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    # mask = (mask > 0.5).astype(np.float32)
    mask = mask.astype(np.float32)
    if (rgb.shape[0] == rgb.shape[1]) and (rgb.shape[0]==512):
        interpolation = get_smart_interpolation(rgb.shape[:2], (mask.shape[0], mask.shape[1]))
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=interpolation)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(rgb, mask, aspect_standard, enlarge_ratio)
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training
    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], 
                                                                        aspect_standard=aspect_standard,
                                                                        tgt_size=render_tgt_size, multiply=multiply)
    
    # 使用智能插值选择
    rgb_interpolation = get_smart_interpolation(rgb.shape[:2], tgt_hw_size)
    mask_interpolation = get_smart_interpolation(mask.shape[:2], tgt_hw_size)
    
    rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=rgb_interpolation)
    mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=mask_interpolation)
    
    if intr is not None:
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
    
    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

    # read shape_param
    shape_param = None
    if get_shape_param:
        cor_flame_path = os.path.join(os.path.dirname(os.path.dirname(rgb_path)),'canonical_flame_param.npz')
        flame_p = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_p['shape'])

    return rgb, mask, intr, shape_param


def extract_imgs_from_video(video_file, save_root, fps):
    print(f"extract_imgs_from_video:{video_file}")
    vr = decord.VideoReader(video_file)
    for i in range(0, len(vr), fps):
        frame = vr[i].asnumpy()
        save_path = os.path.join(save_root, f"{i:05d}.jpg")
        cv2.imwrite(save_path, frame[:, :, (2, 1, 0)])


def predict_motion_seqs_from_images(image_folder:str, save_root, fps=6):
    id_name = os.path.splitext(os.path.basename(image_folder))[0]
    if os.path.isfile(image_folder) and (image_folder.endswith("mp4") or image_folder.endswith("move")):
        save_frame_root = os.path.join(save_root, "extracted_frames", id_name)
        if not os.path.exists(save_frame_root):
            os.makedirs(save_frame_root, exist_ok=True)
            extract_imgs_from_video(video_file=image_folder, save_root=save_frame_root, fps=fps)
        else:
            print("skip extract_imgs_from_video......")
        image_folder = save_frame_root

    image_folder_abspath = os.path.abspath(image_folder)
    print(f"predict motion seq:{image_folder_abspath}")
    save_flame_root = image_folder + "_flame_params_mhmr"
    if not os.path.exists(save_flame_root):
        cmd = f"cd thirdparty/multi-hmr &&  python infer_batch.py  --data_root {image_folder_abspath}  --out_folder {image_folder_abspath} --crop_head   --crop_hand   --pad_ratio 0.2 --smplify"
        os.system(cmd)
    else:
        print("skip predict flame.........")
    return save_flame_root, image_folder


def render_flame_mesh(data, render_intrs, c2ws, human_model_path="./model_zoo/human_parametric_models"):
    from FastAvatar.models.rendering.flame_model.flame import FlameHead, FlameHeadSubdivided
    from FastAvatar.models.rendering.utils.vis_utils import render_mesh

    subdivide = 2
    flame_sub_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=True,
        add_shoulder=False,
        flame_model_path='model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl',
        flame_lmk_embedding_path="model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path="model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj",
        flame_parts_path="model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl",
        subdivide_num=subdivide
    ).cuda()

    shape = data['betas'].to('cuda')
    flame_param = {}
    flame_param['expr'] = data['expr'].to('cuda')
    flame_param['rotation'] = data['rotation'].to('cuda')
    flame_param['neck'] = data['neck_pose'].to('cuda')
    flame_param['jaw'] = data['jaw_pose'].to('cuda')
    flame_param['eyes'] = data['eyes_pose'].to('cuda')
    flame_param['translation'] = data['translation'].to('cuda')
    
    v_cano = flame_sub_model.get_cano_verts(
        shape.unsqueeze(0)
    )

    ret = flame_sub_model.animation_forward(
        v_cano.repeat(flame_param['expr'].shape[0], 1, 1),
        shape.unsqueeze(0).repeat(flame_param['expr'].shape[0], 1),
        flame_param['expr'],
        flame_param['rotation'],
        flame_param['neck'],
        flame_param['jaw'],
        flame_param['eyes'],
        flame_param['translation'],
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        # static_offset=batch_data['static_offset'].to('cuda'),
        static_offset=None,
    )

    flame_face = flame_sub_model.faces.cpu().squeeze().numpy()
    mesh_render_list = []
    num_view = flame_param['expr'].shape[0]
    for v_idx in range(num_view):
        intr = render_intrs[v_idx]
        cam_param = {"focal": torch.tensor([intr[0, 0], intr[1, 1]]), 
                    "princpt": torch.tensor([intr[0, 2], intr[1, 2]])}
        render_shape = int(cam_param['princpt'][1]* 2), int(cam_param['princpt'][0] * 2)     # require h, w

        vertices = ret["animated"][v_idx].cpu().squeeze()

        c2w = c2ws[v_idx]
        w2c = torch.inverse(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        vertices = vertices @ R + T

        mesh_render, is_bkg = render_mesh(vertices,
                                flame_face, cam_param,
                                np.ones((render_shape[0],render_shape[1], 3), dtype=np.float32)*255,
                                return_bg_mask=True)
        mesh_render = mesh_render.astype(np.uint8)
        mesh_render_list.append(mesh_render)
    mesh_render = np.stack(mesh_render_list)
    return mesh_render


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


def load_flame_params(flame_file_path, teeth_bs=None):
    flame_param = dict(np.load(flame_file_path, allow_pickle=True))
    flame_param_tensor = {}
    flame_param_tensor['expr'] = torch.FloatTensor(flame_param['expr'])[0]
    flame_param_tensor['rotation'] = torch.FloatTensor(flame_param['rotation'])[0]
    flame_param_tensor['neck_pose'] = torch.FloatTensor(flame_param['neck_pose'])[0]
    flame_param_tensor['jaw_pose'] = torch.FloatTensor(flame_param['jaw_pose'])[0]
    flame_param_tensor['eyes_pose'] = torch.FloatTensor(flame_param['eyes_pose'])[0]
    flame_param_tensor['translation'] = torch.FloatTensor(flame_param['translation'])[0]
    if teeth_bs is not None:
        flame_param_tensor['teeth_bs'] = torch.FloatTensor(teeth_bs)

    return flame_param_tensor


def prepare_motion_seqs(motion_seqs_dir, image_folder, save_root, fps,
                        bg_color, vis_motion=False, shape_param=None, test_sample=False):
    if motion_seqs_dir is None:
        assert image_folder is not None
        motion_seqs_dir, image_folder = predict_motion_seqs_from_images(image_folder, save_root, fps)
    
    # source images
    c2ws, intrs, bg_colors = [], [], []
    flame_params = []

    # read shape_param
    if shape_param is None:
        print("using driven shape params")
        cor_flame_path = os.path.join(os.path.dirname(motion_seqs_dir),'canonical_flame_param.npz')
        flame_p = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_p['shape'])

    transforms_json = os.path.join(os.path.dirname(motion_seqs_dir), f"transforms.json")
    with open(transforms_json) as fp:
        data = json.load(fp)  
    all_frames = data["frames"]
    all_frames = sorted(all_frames, key=lambda x: x["flame_param_path"])
    print(f"len motion_seq:{len(all_frames)}")
    frame_ids = np.array(list(range(len(all_frames))))
    if test_sample:
        print("sub sample 50 frames for testing.")
        sample_num = 50
        frame_ids = frame_ids[np.linspace(0, frame_ids.shape[0]-1, sample_num).astype(np.int32)]
        print("sub sample ids:", frame_ids)

    teeth_bs_pth = os.path.join(os.path.dirname(motion_seqs_dir), "tracked_teeth_bs.npz")
    if os.path.exists(teeth_bs_pth):
        teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
    else:
        teeth_bs_lst = None

    for idx, frame_id in enumerate(frame_ids):
        frame_info = all_frames[frame_id]
        flame_path = os.path.join(os.path.dirname(motion_seqs_dir), frame_info["flame_param_path"])

        if image_folder is not None:
            file_name = os.path.splitext(os.path.basename(flame_path))[0]
            frame_path = os.path.join(image_folder, file_name + ".png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(image_folder, file_name + ".jpg")
                    
        teeth_bs = teeth_bs_lst[frame_id] if (teeth_bs_lst is not None and len(teeth_bs_lst) > frame_id) else None
        flame_param = load_flame_params(flame_path, teeth_bs)

        c2w, intrinsic = _load_pose(frame_info)
        intrinsic = scale_intrs(intrinsic, 0.5, 0.5)

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        flame_params.append(flame_param)

    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    flame_params_tmp = defaultdict(list)
    for flame in flame_params:
        for k, v in flame.items():
            flame_params_tmp[k].append(v)
    for k, v in flame_params_tmp.items():
        flame_params_tmp[k] = torch.stack(v)
    flame_params = flame_params_tmp
    flame_params["betas"] = shape_param

    if vis_motion:
        motion_render = render_flame_mesh(flame_params, intrs, c2ws)
    else:
        motion_render = None

    # add batch dim
    for k, v in flame_params.items():
        flame_params[k] = v.unsqueeze(0)
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    
    motion_seqs = {}
    motion_seqs["c2ws"] = c2ws
    motion_seqs["intrs"] = intrs
    motion_seqs["bg_colors"] = bg_colors
    motion_seqs["flame_params"] = flame_params
    motion_seqs["vis_motion_render"] = motion_render
    return motion_seqs


def get_smart_interpolation(src_size, dst_size):
    if src_size[0] < dst_size[0] or src_size[1] < dst_size[1]:
        return cv2.INTER_CUBIC
    else:
        return cv2.INTER_AREA