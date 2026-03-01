import os
import argparse
import numpy as np
import torch
import cv2
import json
from tqdm import tqdm
from vhap.model.flame import FlameHead
from vhap.util.render_nvdiffrast import NVDiffRenderer
import glob

# NVDiffRenderer for FLAME mesh rendering

def load_processed_data(frame_dir):
    """Load processed data from a frame directory"""
    processed_data_dir = os.path.join(frame_dir, 'processed_data')
    
    # Load RGB image
    rgb_path = os.path.join(processed_data_dir, 'rgb.npy')
    rgb = np.load(rgb_path)  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Load landmarks
    landmark_path = os.path.join(processed_data_dir, 'landmark2d.npz')
    landmark_data = np.load(landmark_path)
    landmarks = landmark_data['face_landmark_2d'][0]  # (68, 2)

    # Load camera intrinsics
    intrs_path = os.path.join(processed_data_dir, 'intrs.npy')
    intrs = np.load(intrs_path)  # (4, 4)

    # Load FLAME parameters
    flame_param_path = os.path.join(frame_dir, 'flame_param', '00000.npz')
    flame_params = np.load(flame_param_path, allow_pickle=True)
    flame_params = {key: flame_params[key] for key in flame_params.keys()}

    return rgb, landmarks, intrs, flame_params

def load_frame_info(frame_dir):
    # 加载transforms.json，返回frame_info
    transforms_path = os.path.join(frame_dir, 'transforms.json')
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    frame_info = data['frames'][0]
    return frame_info

def render_flame_nvdiffrast(vertices, faces, camera_params, renderer):
    """Render FLAME mesh using NVDiffRenderer (from vhap optimization)"""
    # Convert camera parameters to NVDiffRenderer format
    fx = camera_params['fl_x'][0]
    fy = camera_params['fl_y'][0]
    cx = camera_params['cx'][0]
    cy = camera_params['cy'][0]
    h = int(camera_params['h'][0])
    w = int(camera_params['w'][0])

    # Create camera intrinsics in compact format [fx, fy, cx, cy] and add batch dimension
    K = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).unsqueeze(0).cuda()

    # Convert c2w to w2c (world to camera)
    c2w = torch.tensor(camera_params['transform_matrix'][0], dtype=torch.float32)
    # OpenGL to camera coordinates conversion
    c2w[:3, 1:3] *= -1  # Flip Y and Z axes for OpenGL convention

    # Convert c2w to w2c (camera to world -> world to camera)
    w2c = torch.inverse(c2w).unsqueeze(0).cuda()

    # Convert vertices to torch tensor and add batch dimension
    verts = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0).cuda()
    faces_tensor = torch.tensor(faces, dtype=torch.long).cuda()

    # Render using NVDiffRenderer
    with torch.no_grad():
        out_dict = renderer.render_without_texture(verts, faces_tensor, w2c, K, (h, w), background_color=[1.0, 1.0, 1.0])

        # Extract RGB from RGBA and convert to numpy
        rgba = out_dict['rgba'].squeeze(0).cpu().numpy()  # (H, W, 4)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8

    return rgb

def render_flame_mesh(flame_model, flame_params, device, image_shape, frame_dir, renderer):
    """渲染时直接用augmentation后intrs.npy和c2w，保证mesh和图片空间对齐"""
    # Extract parameters
    shape = torch.tensor(flame_params['shape'], dtype=torch.float32).unsqueeze(0).to(device)
    expr = torch.tensor(flame_params['expr'], dtype=torch.float32).to(device)
    rotation = torch.tensor(flame_params['rotation'], dtype=torch.float32).to(device)
    neck = torch.tensor(flame_params['neck_pose'], dtype=torch.float32).to(device)
    jaw = torch.tensor(flame_params['jaw_pose'], dtype=torch.float32).to(device)
    eyes = torch.tensor(flame_params['eyes_pose'], dtype=torch.float32).to(device)
    translation = torch.tensor(flame_params['translation'], dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        output = flame_model(
            shape=shape,
            expr=expr,
            rotation=rotation,
            neck=neck,
            jaw=jaw,
            eyes=eyes,
            translation=translation,
        )
        # FlameHead.forward returns a list: [vertices, landmarks] when return_landmarks=True
        # or just vertices when return_landmarks=False
        if isinstance(output, (list, tuple)):
            vertices = output[0].cpu().numpy()  # First element is vertices
        else:
            vertices = output.cpu().numpy()  # Single return value
        
        # Ensure vertices shape is (N, 3) - remove batch dimension if present
        if vertices.ndim == 3:
            vertices = vertices[0]  # Take first batch: (batch, N, 3) -> (N, 3)
        elif vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Unexpected vertices shape: {vertices.shape}, expected (N, 3) or (1, N, 3)")

    # 加载c2w
    frame_info = load_frame_info(frame_dir)
    c2w = np.array(frame_info["transform_matrix"]).astype(np.float32)
    c2w[:3, 1:3] *= -1  # OpenGL约定

    # 直接读取augmentation后intrs.npy
    processed_data_dir = os.path.join(frame_dir, 'processed_data')
    intrs = np.load(os.path.join(processed_data_dir, 'intrs.npy'))  # (4, 4)
    fx = intrs[0, 0]
    fy = intrs[1, 1]
    cx = intrs[0, 2]
    cy = intrs[1, 2]
    h = image_shape[0]
    w = image_shape[1]
    camera_params = {
        'fl_x': [fx],
        'fl_y': [fy],
        'cx': [cx],
        'cy': [cy],
        'h': [h],
        'w': [w],
        'transform_matrix': [c2w]
    }

    # Get faces - ensure it's numpy array with correct shape
    faces = flame_model.faces.cpu().numpy()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected faces shape: {faces.shape}, expected (M, 3)")
    
    # 渲染
    color = render_flame_nvdiffrast(vertices, faces, camera_params, renderer)
    return color

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmarks on image"""
    image_with_lmks = image.copy()

    for lmk in landmarks:
        x, y = int(lmk[0]), int(lmk[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image_with_lmks, (x, y), radius, color, -1)

    return image_with_lmks

def create_video_from_frames(frames, output_path, fps=30):
    """Create video from list of frames"""
    if not frames:
        print("No frames to create video")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in tqdm(frames, desc="Writing video"):
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")

def load_processed_data_video_tracking(frame_dir, data_path, frame_idx):
    """Load processed data from video_tracking.py structure"""
    # Load RGB image from processed_data/frame_idx/
    rgb_path = os.path.join(frame_dir, 'rgb.npy')
    rgb = np.load(rgb_path)  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    
    # Get image dimensions
    h, w = rgb.shape[:2]
    
    # Convert RGB to BGR for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Load landmarks
    landmark_path = os.path.join(frame_dir, 'landmark2d.npz')
    landmark_data = np.load(landmark_path)
    landmarks = landmark_data['face_landmark_2d'][0]  # (68, 2) or (68, 3), normalized [0, 1]
    
    # Handle landmarks format: video_track.py saves normalized landmarks [0, 1]
    # Convert from normalized coordinates to pixel coordinates
    if landmarks.shape[1] >= 2:
        landmarks_pixel = landmarks[:, :2].copy()
        # Convert from [0, 1] normalized to pixel coordinates
        landmarks_pixel[:, 0] = landmarks_pixel[:, 0] * w
        landmarks_pixel[:, 1] = landmarks_pixel[:, 1] * h
    else:
        landmarks_pixel = landmarks

    # Load camera intrinsics
    intrs_path = os.path.join(frame_dir, 'intrs.npy')
    intrs = np.load(intrs_path)  # (4, 4)

    # Load FLAME parameters from flame_param/frame_idx.npz
    flame_param_path = os.path.join(data_path, 'flame_param', f'{frame_idx:05d}.npz')
    flame_params = np.load(flame_param_path, allow_pickle=True)
    flame_params = {key: flame_params[key] for key in flame_params.keys()}

    return rgb, landmarks_pixel, intrs, flame_params

def load_frame_info_video_tracking(data_path, frame_idx):
    # 加载transforms.json，返回指定帧的frame_info
    transforms_path = os.path.join(data_path, 'transforms.json')
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # 根据frame_idx获取对应的frame_info
    if frame_idx < len(data['frames']):
        frame_info = data['frames'][frame_idx]
    else:
        # 如果frame_idx超出范围，使用最后一帧的信息
        frame_info = data['frames'][-1]
    
    return frame_info

def render_flame_mesh_video_tracking(flame_model, flame_params, device, image_shape, data_path, frame_idx, renderer):
    """渲染时直接用augmentation后intrs.npy和c2w，保证mesh和图片空间对齐"""
    # Extract parameters
    shape = torch.tensor(flame_params['shape'], dtype=torch.float32).unsqueeze(0).to(device)
    expr = torch.tensor(flame_params['expr'], dtype=torch.float32).to(device)
    rotation = torch.tensor(flame_params['rotation'], dtype=torch.float32).to(device)
    neck = torch.tensor(flame_params['neck_pose'], dtype=torch.float32).to(device)
    jaw = torch.tensor(flame_params['jaw_pose'], dtype=torch.float32).to(device)
    eyes = torch.tensor(flame_params['eyes_pose'], dtype=torch.float32).to(device)
    translation = torch.tensor(flame_params['translation'], dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        output = flame_model(
            shape=shape,
            expr=expr,
            rotation=rotation,
            neck=neck,
            jaw=jaw,
            eyes=eyes,
            translation=translation,
        )
        # FlameHead.forward returns a list: [vertices, landmarks] when return_landmarks=True
        # or just vertices when return_landmarks=False
        if isinstance(output, (list, tuple)):
            vertices = output[0].cpu().numpy()  # First element is vertices
        else:
            vertices = output.cpu().numpy()  # Single return value
        
        # Ensure vertices shape is (N, 3) - remove batch dimension if present
        if vertices.ndim == 3:
            vertices = vertices[0]  # Take first batch: (batch, N, 3) -> (N, 3)
        elif vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Unexpected vertices shape: {vertices.shape}, expected (N, 3) or (1, N, 3)")

    # 加载c2w - 使用对应帧的transform_matrix
    frame_info = load_frame_info_video_tracking(data_path, frame_idx)
    c2w = np.array(frame_info["transform_matrix"]).astype(np.float32)
    c2w[:3, 1:3] *= -1  # OpenGL约定

    # 直接读取augmentation后intrs.npy
    processed_data_dir = os.path.join(data_path, 'processed_data', f'{frame_idx:05d}')
    intrs = np.load(os.path.join(processed_data_dir, 'intrs.npy'))  # (4, 4)
    fx = intrs[0, 0]
    fy = intrs[1, 1]
    cx = intrs[0, 2]
    cy = intrs[1, 2]
    h = image_shape[0]
    w = image_shape[1]
    camera_params = {
        'fl_x': [fx],
        'fl_y': [fy],
        'cx': [cx],
        'cy': [cy],
        'h': [h],
        'w': [w],
        'transform_matrix': [c2w]
    }

    # Get faces - ensure it's numpy array with correct shape
    faces = flame_model.faces.cpu().numpy()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected faces shape: {faces.shape}, expected (M, 3)")
    
    # 渲染
    color = render_flame_nvdiffrast(vertices, faces, camera_params, renderer)
    return color

def process_single_camera(cam_id, base_data_path, args, device_id):
    """Process a single camera - this function will be called in parallel"""
    print(f"Processing camera: {cam_id} on device {device_id}")
    
    # Set device for this process
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # Initialize FLAME model for this process
    flame_model = FlameHead(
        300, 100, add_teeth=True,
        flame_model_path=args.flame_model_path,
        flame_lmk_embedding_path=args.flame_lmk_path,
        flame_template_mesh_path=args.flame_template_path,
    ).to(device)

    # Initialize NVDiffRenderer for this process
    renderer = NVDiffRenderer(use_opengl=False, lighting_space='camera')

    # Clear GPU cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # data_path is the directory containing processed_data/ and transforms.json
    data_path = base_data_path
    
    if not os.path.exists(data_path):
        print(f"Data path does not exist: {data_path}")
        return cam_id, None
    
    # Find all frame directories in processed_data
    processed_data_dir = os.path.join(data_path, 'processed_data')
    if not os.path.exists(processed_data_dir):
        print(f"Processed data directory not found: {processed_data_dir}")
        return cam_id, None
    
    frame_dirs = glob.glob(os.path.join(processed_data_dir, '*'))
    frame_dirs = [d for d in frame_dirs if os.path.isdir(d)]
    frame_dirs = sorted(frame_dirs)
    
    if not frame_dirs:
        print(f"No frame directories found in {processed_data_dir}")
        return cam_id, None
    
    print(f"Found {len(frame_dirs)} frame directories for {cam_id}")
    
    # === Camera Parameter Consistency Check ===
    import random
    import numpy as np
    if len(frame_dirs) >= 4:
        sampled_dirs = random.sample(frame_dirs, 4)
    else:
        sampled_dirs = frame_dirs

    intrs_list = []
    c2w_list = []
    idx_list = []

    for frame_dir in sampled_dirs:
        frame_name = os.path.basename(frame_dir)
        frame_idx = int(frame_name)
        idx_list.append(frame_idx)
        # 加载intrinsics
        intrs = np.load(os.path.join(frame_dir, 'intrs.npy'))
        # 加载extrinsics (c2w)
        frame_info = load_frame_info_video_tracking(data_path, frame_idx)
        c2w = np.array(frame_info["transform_matrix"])
        intrs_list.append(intrs)
        c2w_list.append(c2w)

    # 判断是否全部一致
    intrs_ref = intrs_list[0]
    c2w_ref = c2w_list[0]
    intrs_all_equal = all(np.allclose(intrs_ref, intrs, atol=1e-6) for intrs in intrs_list)
    c2w_all_equal = all(np.allclose(c2w_ref, c2w, atol=1e-6) for c2w in c2w_list)

    if intrs_all_equal and c2w_all_equal:
        print(f'[CHECK] Camera intrinsics and extrinsics are CONSISTENT across sampled frames for {cam_id}: {idx_list}')
    else:
        print(f'[CHECK] Camera parameters are NOT consistent across sampled frames for {cam_id}: {idx_list}')
        for i, idx in enumerate(idx_list):
            if not np.allclose(intrs_ref, intrs_list[i], atol=1e-6):
                print(f'  - Intrinsics differ at frame {idx:05d} for {cam_id}')
            if not np.allclose(c2w_ref, c2w_list[i], atol=1e-6):
                print(f'  - Extrinsics differ at frame {idx:05d} for {cam_id}')
    # === End Camera Parameter Consistency Check ===

    # Create output directory (use data_dir name as identifier)
    camera_output_dir = args.output_dir
    os.makedirs(camera_output_dir, exist_ok=True)
    
    # Process frames
    lmk_frames = []
    flame_frames = []
    first_frame_saved = False  # Flag to track if first frame has been saved
    
    for frame_dir in tqdm(frame_dirs, desc=f"Processing frames for {cam_id}"):
        frame_name = os.path.basename(frame_dir)
        try:
            frame_idx = int(frame_name)
        except ValueError:
            print(f"Skipping non-numeric directory: {frame_name}")
            continue
        
        # Load processed data from video_tracking.py structure
        rgb, landmarks, intrs, flame_params = load_processed_data_video_tracking(frame_dir, data_path, frame_idx)

        # Create image with landmarks
        lmk_frame = draw_landmarks(rgb, landmarks)
        lmk_frames.append(lmk_frame)
        
        # Create FLAME mesh overlay
        flame_mesh = render_flame_mesh_video_tracking(flame_model, flame_params, device, rgb.shape, data_path, frame_idx, renderer)
        
        # Blend flame mesh with original image - make gray mesh more visible
        # Create mask from mesh (where mesh is not black/background)
        mesh_mask = (flame_mesh.sum(axis=2) > 10).astype(np.float32)[:, :, None]
        # Use higher alpha for mesh to make gray head more visible
        mesh_alpha = 0.7  # Higher visibility for gray mesh
        # Blend: gray mesh shows prominently where it exists
        flame_overlay = (mesh_alpha * flame_mesh[:, :, :3] * mesh_mask + 
                        (1 - mesh_alpha * mesh_mask) * rgb).astype(np.uint8)
        flame_frames.append(flame_overlay)

        
        # Save first frame images
        if not first_frame_saved:
            try:
                # Save original RGB image (data is already in correct format)
                first_frame_rgb_path = os.path.join(camera_output_dir, 'first_frame_rgb.png')
                cv2.imwrite(first_frame_rgb_path, rgb)
                
                print(f"Saved first frame RGB image")
                first_frame_saved = True
                
            except Exception as e:
                print(f"Failed to save first frame RGB image: {e}")

    
    # Create videos for this camera
    if lmk_frames:
        lmk_video_path = os.path.join(camera_output_dir, 'landmarks.mp4')
        create_video_from_frames(lmk_frames, lmk_video_path, args.fps)
    
    if flame_frames:
        flame_video_path = os.path.join(camera_output_dir, 'flame.mp4')
        create_video_from_frames(flame_frames, flame_video_path, args.fps)
    
    # === Canonical FLAME mesh video ===
    cano_param_path = os.path.join(data_path, 'canonical_flame_param.npz')
    if os.path.exists(cano_param_path):
        cano_flame_params = np.load(cano_param_path, allow_pickle=True)
        cano_flame_params = {key: cano_flame_params[key] for key in cano_flame_params.keys()}

        cano_frames = []
        
        for frame_dir in tqdm(frame_dirs, desc=f"Rendering canonical mesh for {cam_id}"):
            frame_name = os.path.basename(frame_dir)
            try:
                frame_idx = int(frame_name)
            except ValueError:
                continue
            rgb, _, _, _ = load_processed_data_video_tracking(frame_dir, data_path, frame_idx)
            try:
                flame_mesh = render_flame_mesh_video_tracking(
                    flame_model, cano_flame_params, device, rgb.shape, data_path, frame_idx, renderer
                )
                # Use same blending as regular flame mesh for consistency
                mesh_mask = (flame_mesh.sum(axis=2) > 10).astype(np.float32)[:, :, None]
                mesh_alpha = 0.7  # Higher visibility for gray mesh
                flame_overlay = (mesh_alpha * flame_mesh[:, :, :3] * mesh_mask + 
                                (1 - mesh_alpha * mesh_mask) * rgb).astype(np.uint8)
                cano_frames.append(flame_overlay)
                        
            except Exception as e:
                cano_frames.append(rgb)

        if cano_frames:
            cano_video_path = os.path.join(camera_output_dir, 'canonical_flame.mp4')
            create_video_from_frames(cano_frames, cano_video_path, args.fps)
    # === End Canonical FLAME mesh video ===
    
    # Clear GPU cache after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Finished processing camera: {cam_id}")
    return cam_id, camera_output_dir

def main():
    parser = argparse.ArgumentParser(description='Check processed data and generate visualization videos')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing processed_data/ and transforms.json (video_track.py output)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for videos (default: data_dir/../check_output)')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for output videos')
    parser.add_argument('--flame_model_path', type=str, 
                        default='model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl')
    parser.add_argument('--flame_lmk_path', type=str, 
                        default='model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy')
    parser.add_argument('--flame_template_path', type=str, 
                        default='model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: number of available GPUs or 4 for single GPU)')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Force single GPU mode for parallel processing')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # Check required files/directories
    processed_data_dir = os.path.join(args.data_dir, 'processed_data')
    transforms_path = os.path.join(args.data_dir, 'transforms.json')
    
    if not os.path.exists(processed_data_dir):
        print(f"Error: processed_data/ directory not found in {args.data_dir}")
        return
    
    if not os.path.exists(transforms_path):
        print(f"Error: transforms.json not found in {args.data_dir}")
        return
    
    print(f"Processing data directory: {args.data_dir}")
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        # Use parent directory of data_dir with a check_output subdirectory
        parent_dir = os.path.dirname(os.path.abspath(args.data_dir))
        data_dir_name = os.path.basename(os.path.abspath(args.data_dir))
        args.output_dir = os.path.join(parent_dir, 'check_output', data_dir_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process single camera directory
    base_data_path = args.data_dir
    cam_id = 'output'  # Simple identifier
    
    # Process the single camera
    print("Processing data...")
    result = process_single_camera(cam_id, base_data_path, args, 0)
    
    # Print summary
    cam_id_result, output_dir = result
    if output_dir is not None:
        print(f"\nProcessing completed successfully!")
        print(f"All videos saved to: {output_dir}")
    else:
        print(f"\nProcessing failed!")

if __name__ == "__main__":
    main() 