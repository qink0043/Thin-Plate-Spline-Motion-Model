import sys
import yaml
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio.v2 as imageio
import cv2
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
import mediapipe as mp
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

def relative_kp(kp_source, kp_driving, kp_driving_initial, kp_scale=1.0):
    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = (np.sqrt(source_area) / np.sqrt(driving_area)) * kp_scale

    kp_new = {k: v for k, v in kp_driving.items()}
    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']
    return kp_new

def resize_with_padding(image, size):
    """Resizes image maintaining aspect ratio and adds black padding."""
    h, w = image.shape[:2]
    target_h, target_w = size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Use cv2 for faster resize and better alpha handling
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return padded

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network

def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode='relative', kp_scale=1.0, avd_blend=1.0, source_alpha=None):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        
        # Prepare alpha for warping if provided
        alpha = None
        if source_alpha is not None:
            alpha = torch.tensor(source_alpha[np.newaxis, np.newaxis].astype(np.float32)).to(device)

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2]), desc=f"TPS Animating Character ({mode})"):
            driving_frame = driving[:, :, frame_idx].to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=kp_driving_initial, kp_scale=kp_scale)
            elif mode == 'avd':
                kp_avd = avd_network(kp_source, kp_driving)
                if avd_blend < 1.0:
                    kp_rel = relative_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=kp_driving_initial, kp_scale=kp_scale)
                    kp_norm = {k: v for k, v in kp_avd.items()}
                    # Linearly interpolate keypoints to balance identity and motion
                    kp_norm['fg_kp'] = (1.0 - avd_blend) * kp_rel['fg_kp'] + avd_blend * kp_avd['fg_kp']
                else:
                    kp_norm = kp_avd
                
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param=None, dropout_flag=False)
            out = inpainting_network(source, dense_motion)
            
            # Final processed frame
            gen_frame = out['prediction']
            
            # Background Compositing
            if alpha is not None:
                # 1. Warp the source alpha onto the driving pose
                deformation = dense_motion['deformation']
                warped_alpha = inpainting_network.deform_input(alpha, deformation)
                
                # 2. Use the generated frame's intensity as a base for the mask
                # Since the background in the raw generation is black, this is very accurate
                intensity_mask = torch.max(gen_frame, dim=1, keepdim=True)[0]
                intensity_mask = torch.clamp(intensity_mask * 10.0, 0, 1)
                
                # 3. Combine both: must be in the warped alpha AND have pixels
                mask = warped_alpha * intensity_mask
                
                # Sharpen and dilate slightly to ensure it covers the character edges
                mask = torch.clamp(mask * 2.0, 0, 1) 
                mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                
                # Smooth the mask edges slightly
                mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)

                # Combine generated character with original driving background
                bg_frame = driving_frame
                gen_frame = gen_frame * mask + bg_frame * (1 - mask)

            predictions.append(np.transpose(gen_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            
    return predictions

def get_user_points(image_rgb, window_name="Select 2 Points", num_points=2):
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < num_points:
                points.append([x, y])
                # Draw a point on the image
                cv2.circle(img_bgr, (x, y), 3, (0, 0, 255), -1)
                # Draw line if 2 points
                if len(points) == 2:
                    cv2.line(img_bgr, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
                cv2.imshow(window_name, img_bgr)
                
    # Convert RGB to BGR for OpenCV display
    if image_rgb.shape[2] == 4:
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGRA)
    else:
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
    cv2.imshow(window_name, img_bgr)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"\n[{window_name}]:")
    print(f"-> Please use your mouse to click {num_points} points (e.g., top and bottom of the cudgel/stick).")
    print("-> After clicking, press ANY KEY on your keyboard to continue.")
    
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    
    if len(points) < num_points:
        raise ValueError(f"Did not select {num_points} points! Aborting.")
    return points

def get_similarity_transform(src_pts, dst_pts, fixed_scale=None):
    """Calculates Similarity Transform Matrix M from 2 points (translation, rotation, uniform scale)."""
    p1, p2 = src_pts[0], src_pts[1]
    q1, q2 = dst_pts[0], dst_pts[1]
    
    dx_p = p2[0] - p1[0]
    dy_p = p2[1] - p1[1]
    dx_q = q2[0] - q1[0]
    dy_q = q2[1] - q1[1]
    
    current_scale = np.hypot(dx_q, dy_q) / (np.hypot(dx_p, dy_p) + 1e-6)
    scale = fixed_scale if fixed_scale is not None else current_scale
    
    angle_p = np.arctan2(dy_p, dx_p)
    angle_q = np.arctan2(dy_q, dx_q)
    angle = angle_q - angle_p
    
    alpha = scale * np.cos(angle)
    beta = scale * np.sin(angle)
    
    # Use midpoint q for translation if scale is fixed to ensure it stays centered
    mid_p = (np.array(p1) + np.array(p2)) / 2
    mid_q = (np.array(q1) + np.array(q2)) / 2
    
    tx = mid_q[0] - (alpha * mid_p[0] - beta * mid_p[1])
    ty = mid_q[1] - (beta * mid_p[0] + alpha * mid_p[1])
    
    M = np.array([
        [alpha, -beta, tx],
        [beta,   alpha, ty]
    ])
    return M

if __name__ == "__main__":
    parser = ArgumentParser(description="TPS Animation with Object Tracking (Golden Cudgel/Stick)")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")
    
    parser.add_argument("--source_character", required=True, help="path to source character image (NO cudgel)")
    parser.add_argument("--source_cudgel", required=True, help="path to source cudgel ONLY image (Transparent PNG)")
    parser.add_argument("--driving_video", required=True, help="path to driving video")
    parser.add_argument("--result_video", default='./result_with_cudgel.mp4', help="path to output")
    
    parser.add_argument("--src_pts", type=str, default=None, help="Comma separated x1,y1,x2,y2 for source cudgel. Skips GUI if provided.")
    parser.add_argument("--dst_pts", type=str, default=None, help="Comma separated x1,y1,x2,y2 for driving video first frame. Skips GUI if provided.")
    parser.add_argument("--save_first_frame", type=str, default=None, help="Save first frame of driving video for manual coordinate finding")
    parser.add_argument("--fixed_scale", action="store_true", help="Keep cudgel scale fixed based on the first frame (previves stretching)")
    parser.add_argument("--smooth_points", type=int, default=0, help="Window size for temporal smoothing of tracked points (e.g. 3 or 5)")
    parser.add_argument("--debug_tracking", type=str, default=None, help="Path to save a debug video showing the tracked points")
    
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))))
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'])
    parser.add_argument("--cpu", action="store_true", help="cpu mode")
    parser.add_argument("--use_pose", action="store_true", help="Use Mediapipe Pose to track wrists instead of optical flow (Recommended)")
    parser.add_argument("--use_color", action="store_true", help="Use HSV Color tracking for the golden cudgel (Excellent fallback)")
    parser.add_argument("--hsv_range", type=str, default=None, help="Manual H,S,V Lower/Upper (e.g. 15,100,100,45,255,255)")
    parser.add_argument("--hsv_tolerance", type=str, default="20,100,100", help="H,S,V tolerance (wider for more yellow types)")
    parser.add_argument("--debug_mask", type=str, default=None, help="Path to save the HSV mask debug video")
    parser.add_argument("--sample_radius", type=int, default=5, help="Radius for sampling color around points")
    parser.add_argument("--annotations", type=str, default=None, help="Path to pre-recorded JSON annotations for the cudgel")
    parser.add_argument("--padding", action="store_true", help="Add padding to maintain aspect ratio (prevents squashing)")
    parser.add_argument("--kp_scale", type=float, default=1.0, help="Manual scale multiplier for relative keypoint movement")
    parser.add_argument("--avd_blend", type=float, default=1.0, help="Blend factor between Relative (0.0) and AVD (1.0). Useful for balancing identity (Relative) and motion (AVD).")
    parser.add_argument("--use_driving_bg", action="store_true", help="Composite the generated character onto the original driving video background (requires transparent source character).")

    
    opt = parser.parse_args()
    
    if 'vox' in opt.config and 'taichi' in opt.checkpoint:
        print("\n[WARNING] Configuration mismatch! You are using 'vox' config with 'taichi' checkpoint.")
        print("Please use --config config/taichi-256.yaml for better results.\n")

    device = torch.device('cpu' if opt.cpu else 'cuda')

    print("Loading resources...")
    source_character = imageio.imread(opt.source_character)
    source_cudgel = imageio.imread(opt.source_cudgel)  # Should have alpha channel
    print(f"DEBUG: source_character shape: {source_character.shape}")
    print(f"DEBUG: source_cudgel shape: {source_cudgel.shape}")
    
    if source_cudgel.shape[2] != 4:
        raise ValueError("The source_cudgel image MUST have an alpha channel (Transparent PNG).")

    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    for im in reader:
        driving_video.append(im)
    reader.close()
    
    # Resize to the model's expected shape
    H, W = opt.img_shape
    
    # Pre-process background: If background is white and alpha is solid, force background to black
    # This ensures the model generates against black, which makes mask extraction robust.
    is_white_bg = np.all(source_character[0, 0, :3] > 240) # Check top-left corner
    is_full_alpha = source_character.shape[2] == 4 and np.all(source_character[..., 3] > 250)
    
    source_alpha = None
    if source_character.shape[2] == 4 and not is_full_alpha:
        source_alpha = source_character[..., 3] / 255.0
        print("Detected functional alpha channel in source character.")
    elif is_white_bg:
        # Create alpha from color (Keying out white)
        diff = np.abs(source_character[..., :3].astype(float) - 255.0).sum(axis=-1)
        source_alpha = (diff > 10).astype(float) # Any pixel not white
        source_character[diff <= 10, :3] = 0 # Force white background to black
        print("Detected white background. Forced to black for better masking.")
    elif source_character.shape[2] == 4:
        # It's 4 channel but alpha is solid? Fall back to simple intensity or treat as opaque
        print("Source character has solid alpha channel. Using color intensity as mask fallback.")
        source_alpha = np.ones(source_character.shape[:2], dtype=float) # Will be refined later

    if opt.padding:
        source_character_resized = resize_with_padding(source_character, opt.img_shape)
        source_character = source_character_resized[..., :3]
        if source_alpha is not None:
             source_alpha = resize_with_padding(source_alpha[..., np.newaxis], opt.img_shape)[..., 0]
        
        # Keep alpha channel for cudgel, then move to RGB for prediction if needed
        # Actually source_cudgel_rgba is for overlay, so keep it high quality
        source_cudgel_rgba = resize_with_padding(source_cudgel, opt.img_shape)
        driving_video = [resize_with_padding(frame, opt.img_shape)[..., :3] for frame in driving_video]
        print(f"Resized inputs with PADDING to {opt.img_shape} (preserving aspect ratio).")
    else:
        source_character_resized = resize(source_character, opt.img_shape)
        source_character = source_character_resized[..., :3]
        if source_alpha is not None:
            source_alpha = resize(source_alpha[..., np.newaxis], opt.img_shape)[..., 0]

        # Keep alpha channel for cudgel, resize using cv2 to avoid alpha blending background issues
        source_cudgel_rgba = cv2.resize(source_cudgel, (W, H), interpolation=cv2.INTER_AREA) 
        driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
        print(f"Resized inputs with SQUASHING to {opt.img_shape}.")
    
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint, device=device)
    
    print("\n--- STEP 1 & 2: Define Cudgel Points ---")
    # Multiplying by 255 because skimage resize returns float [0, 1]
    first_dr_frame = (driving_video[0] * 255).astype(np.uint8)

    if opt.save_first_frame:
        imageio.imwrite(opt.save_first_frame, first_dr_frame)
        print(f"Saved first frame of driving video to {opt.save_first_frame}.")
        print("Please check the visual coordinates and run again.")
        sys.exit(0)

    if opt.src_pts:
        s_pts = list(map(int, opt.src_pts.split(',')))
        src_pts = [[s_pts[0], s_pts[1]], [s_pts[2], s_pts[3]]]
        print(f"Using provided source coordinates: {src_pts}")
    else:
        # Auto-detect from alpha channel to avoid GUI in headless environments
        alpha = source_cudgel_rgba[:, :, 3]
        pts = np.argwhere(alpha > 0)
        if len(pts) > 0:
            # Find the two points furthest apart in the alpha mask (usually the ends of the stick)
            from scipy.spatial import distance
            dists = distance.cdist(pts, pts, 'euclidean')
            idx = np.unravel_index(dists.argmax(), dists.shape)
            p1, p2 = pts[idx[0]], pts[idx[1]]
            src_pts = [[int(p1[1]), int(p1[0])], [int(p2[1]), int(p2[0])]]
            print(f"Auto-detected source points from alpha mask: {src_pts}")
        else:
            try:
                src_pts = get_user_points((source_cudgel_rgba).astype(np.uint8), "Select 2 points on SOURCE cudgel")
            except Exception as e:
                print(f"\n[ERROR] Display connection failed: {e}")
                print("Headless environment detected. Please provide --src_pts x1,y1,x2,y2 manually.")
                sys.exit(1)

    if opt.dst_pts:
        d_pts = list(map(int, opt.dst_pts.split(',')))
        dst_pts = [[d_pts[0], d_pts[1]], [d_pts[2], d_pts[3]]]
        print(f"Using provided destination coordinates: {dst_pts}")
    elif opt.annotations and os.path.exists(opt.annotations):
        with open(opt.annotations, 'r') as f:
            ann_data = json.load(f)
            if len(ann_data) > 0:
                dst_pts = ann_data[0]
                print(f"Using first frame of '{opt.annotations}' for initial destination points: {dst_pts}")
            else:
                try:
                    dst_pts = get_user_points(first_dr_frame, "Select 2 points on DRIVING video")
                except Exception as e:
                    print(f"\n[ERROR] Display connection failed: {e}")
                    print("Headless environment detected. Annotation file is empty or missing. Please provide --dst_pts x1,y1,x2,y2 manually.")
                    sys.exit(1)
    else:
        try:
            dst_pts = get_user_points(first_dr_frame, "Select exactly the same 2 points on DRIVING video")
        except Exception as e:
            print(f"\n[ERROR] Display connection failed: {e}")
            print("Headless environment detected. Please provide --dst_pts x1,y1,x2,y2 manually or use --annotations.")
            sys.exit(1)

    # --- Step 3: TPS Character Animation ---
    print("\n--- STEP 3: Generating TPS Animation for Character... ---")
    if opt.use_driving_bg and source_alpha is None:
        print("WARNING: --use_driving_bg was set but source_character has no alpha channel. Background replacement will be skipped.")
    
    predictions = make_animation(source_character, driving_video, inpainting, kp_detector, 
                                 dense_motion_network, avd_network, device=device, mode=opt.mode,
                                 kp_scale=opt.kp_scale, avd_blend=opt.avd_blend,
                                 source_alpha=source_alpha if opt.use_driving_bg else None)

    # --- Step 4: Track Cudgel ---
    tracked_points_list = []
    debug_frames = []

    if opt.annotations and os.path.exists(opt.annotations):
        print(f"\n--- STEP 4: Loading Manual Annotations from {opt.annotations}... ---")
        with open(opt.annotations, 'r') as f:
            tracked_points_list = json.load(f)
        
        if len(tracked_points_list) != len(driving_video):
            print(f"Warning: Annotation length ({len(tracked_points_list)}) does not match video length ({len(driving_video)}).")
            if len(tracked_points_list) < len(driving_video):
                print("Padding with last known position...")
                while len(tracked_points_list) < len(driving_video):
                    tracked_points_list.append(tracked_points_list[-1])
        
        if len(tracked_points_list) > 0:
            dst_pts = tracked_points_list[0]
    else:
        # --- Automatic Tracking Methods ---
        tracked_points_list = [dst_pts]
        
        if opt.use_pose:
            print("\n--- STEP 4: Tracking Cudgel in Driving Video (Mediapipe Pose)... ---")
            try:
                from mediapipe.python.solutions import pose as mp_pose
            except ImportError:
                try:
                    import mediapipe as mp
                    mp_pose = mp.solutions.pose
                except Exception:
                    print("Warning: Mediapipe Pose not found. Falling back to next method.")
                    opt.use_pose = False
            
            if opt.use_pose:
                pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
                res = pose.process(cv2.cvtColor(first_dr_frame, cv2.COLOR_RGB2BGR))
                if not res.pose_landmarks:
                     print("Warning: Pose not detected in first frame. Falling back.")
                     opt.use_pose = False
                else:
                    def get_wrists(landmarks, h, w):
                        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                        return np.array([[l_wrist.x * w, l_wrist.y * h], [r_wrist.x * w, r_wrist.y * h]])

                    wrists0 = get_wrists(res.pose_landmarks.landmark, H, W)
                    last_wrists = wrists0
                    for i in tqdm(range(1, len(driving_video)), desc="Pose Tracking"):
                        frame_bgr = (driving_video[i] * 255).astype(np.uint8)
                        res = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))
                        if res.pose_landmarks:
                            wrists_curr = get_wrists(res.pose_landmarks.landmark, H, W)
                            M_move = get_similarity_transform(wrists0, wrists_curr)
                            p1_hom, p2_hom = np.array([dst_pts[0][0], dst_pts[0][1], 1]), np.array([dst_pts[1][0], dst_pts[1][1], 1])
                            new_p1, new_p2 = M_move @ p1_hom, M_move @ p2_hom
                            tracked_points_list.append([new_p1.tolist(), new_p2.tolist()])
                        else:
                            tracked_points_list.append(tracked_points_list[-1])
                        if opt.debug_tracking:
                            dbg_frame = frame_bgr.copy()
                            pts = tracked_points_list[-1]
                            cv2.circle(dbg_frame, tuple(map(int, pts[0])), 4, (255, 0, 0), -1)
                            cv2.circle(dbg_frame, tuple(map(int, pts[1])), 4, (0, 255, 0), -1)
                            cv2.putText(dbg_frame, f"Pose Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            debug_frames.append(dbg_frame)
                    pose.close()

        if opt.use_color and not opt.use_pose:
            print("\n--- STEP 4: Tracking Cudgel in Driving Video (HSV Color Tracking)... ---")
            debug_mask_frames = []
            
            # Auto-Calibration or Manual Range
            if opt.hsv_range:
                hsv_vals = list(map(int, opt.hsv_range.split(',')))
                lower_hsv = np.array(hsv_vals[:3])
                upper_hsv = np.array(hsv_vals[3:])
                print(f"Using Manual HSV Range: Lower={lower_hsv}, Upper={upper_hsv}")
            else:
                # Auto-calibrate: Median-based sampling
                print(f"Auto-calibrating color from selected points (Radius={opt.sample_radius} window)...")
                hsv_0 = cv2.cvtColor(first_dr_frame, cv2.COLOR_RGB2HSV)
                
                def get_percentile_hsv(img, x, y, r, name):
                    x, y = int(x), int(y)
                    row_start, row_end = max(0, y-r), min(H, y+r+1)
                    col_start, col_end = max(0, x-r), min(W, x+r+1)
                    
                    # Check if the ROI is empty
                    if row_end <= row_start or col_end <= col_start:
                        print(f"  Warning: Sampling ROI is empty at ({x}, {y})")
                        return np.array([0, 0, 0]), np.array([255, 255, 255])
                    
                    roi_hsv = img[row_start:row_end, col_start:col_end]
                    roi_rgb = first_dr_frame[row_start:row_end, col_start:col_end]
                    
                    imageio.imwrite(f"{name}.png", (resize(roi_rgb, (64, 64)) * 255).astype(np.uint8))
                    print(f"  Saved sampled patch to {name}.png (Enlarged 64x64). Check if this is the stick color!")
                    
                    valid = roi_hsv[roi_hsv[:, :, 2] > 30] # Ignore dark pixels
                    if len(valid) == 0: valid = roi_hsv.reshape(-1, 3)
                    
                    # Take 10th and 90th percentiles to define the range
                    low = np.percentile(valid, 10, axis=0)
                    high = np.percentile(valid, 90, axis=0)
                    return low, high

                l1, h1 = get_percentile_hsv(hsv_0, dst_pts[0][0], dst_pts[0][1], opt.sample_radius, "sample_pt1")
                l2, h2 = get_percentile_hsv(hsv_0, dst_pts[1][0], dst_pts[1][1], opt.sample_radius, "sample_pt2")
                
                lower_hsv = np.minimum(l1, l2)
                upper_hsv = np.maximum(h1, h2)
                
                # Add a bit of extra tolerance
                tol = list(map(int, opt.hsv_tolerance.split(',')))
                lower_hsv = np.clip(lower_hsv - np.array(tol), 0, 255).astype(np.uint8)
                upper_hsv = np.clip(upper_hsv + np.array(tol), 0, 255).astype(np.uint8)
                
                # Saturation Floor: Gray cement floors typically have low saturation (< 40)
                # The yellow stick should be vibrant.
                lower_hsv[1] = max(lower_hsv[1], 40) # Relaxed slightly so dark parts of stick aren't lost
                lower_hsv[2] = max(lower_hsv[2], 40) 
                
                print(f"Locked HSV Range (after floor): Lower={lower_hsv}, Upper={upper_hsv}")

            # Initialize tracking state
            prev_p0, prev_p1 = np.array(dst_pts[0]), np.array(dst_pts[1])
            prev_center = (prev_p0 + prev_p1) / 2
            velocity = np.array([0.0, 0.0])
            max_observed_length = np.linalg.norm(prev_p1 - prev_p0)
            roi_radius = max(80, int(max_observed_length * 0.8))
            frames_lost = 0

            for i in tqdm(range(1, len(driving_video)), desc="Color Tracking"):
                frame_bgr = (driving_video[i] * 255).astype(np.uint8)
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV)
                
                # 1. Base Mask
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                
                # 2. Dynamic ROI Prediction
                # Predict center based on velocity
                predicted_center = (prev_center + velocity).astype(np.int32)
                
                # Expand ROI if lost
                current_roi_radius = roi_radius + (frames_lost * 20)
                
                roi_mask = np.zeros_like(mask)
                cv2.circle(roi_mask, tuple(predicted_center), current_roi_radius, 255, -1)
                mask = cv2.bitwise_and(mask, roi_mask)
                
                # Pre-cleaning: remove small noise
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                
                # Post-cleaning: fill gaps and solidify (aggressive to fix "half stick")
                # Using a larger closing kernel to bridge fingers
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                _, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
                kernel_close = np.ones((21, 21), np.uint8) # Increased size
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1) 
                
                if opt.debug_mask:
                    debug_mask_frames.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_hull = None
                if contours:
                    valid_candidates = []
                    frame_area = W * H
                    
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < 10 or area > (frame_area * 0.2): continue
                        
                        hull = cv2.convexHull(c)
                        rect = cv2.minAreaRect(hull)
                        (cx, cy), (w, h), angle = rect
                        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                        
                        hull_center = np.mean(hull, axis=0)[0]
                        dist_from_predicted = np.linalg.norm(hull_center - predicted_center)
                        
                        # Target aspect ratio: sticks are long
                        if aspect_ratio > 1.8:
                            # Score: balance proximity, size, and shape
                            # We penalize distance less than before to allow fast movement
                            score = (area * aspect_ratio) / (dist_from_predicted + 10.0)
                            valid_candidates.append((hull, score, (cx, cy), max(w, h)))
                    
                    if valid_candidates:
                        best_cand = max(valid_candidates, key=lambda x: x[1])
                        best_hull = best_cand[0]
                        curr_center = np.array(best_cand[2])
                        curr_len = best_cand[3]
                        
                        # Update tracking state
                        velocity = curr_center - prev_center
                        prev_center = curr_center
                        max_observed_length = max(max_observed_length, curr_len)
                        frames_lost = 0
                        
                        # Target points: furthest points in the hull
                        dist_max = 0
                        pts_max = [tracked_points_list[-1][0], tracked_points_list[-1][1]]
                        c_sub = best_hull[::max(1, len(best_hull)//30)]
                        for p_i in range(len(c_sub)):
                            for p_j in range(p_i + 1, len(c_sub)):
                                d = np.linalg.norm(c_sub[p_i][0] - c_sub[p_j][0])
                                if d > dist_max:
                                    dist_max = d
                                    pts_max = [c_sub[p_i][0].astype(float), c_sub[p_j][0].astype(float)]
                        
                        # --- Occlusion Handling ---
                        # If the detected length is much shorter than max_observed_length, 
                        # it's likely occluded by a hand. Extrapolate endpoints.
                        if dist_max < (max_observed_length * 0.85):
                            direction = (pts_max[1] - pts_max[0]) / (dist_max + 1e-6)
                            midpoint = (pts_max[0] + pts_max[1]) / 2
                            # Extend from center to match expected length
                            pts_max[0] = midpoint - direction * (max_observed_length / 2)
                            pts_max[1] = midpoint + direction * (max_observed_length / 2)
                        
                        # Ensure continuity (swap points if they flipped)
                        old_p0 = np.array(tracked_points_list[-1][0])
                        if np.linalg.norm(pts_max[0] - old_p0) > np.linalg.norm(pts_max[1] - old_p0):
                            pts_max = [pts_max[1], pts_max[0]]
                        
                        tracked_points_list.append([pts_max[0].tolist(), pts_max[1].tolist()])
                    else:
                        frames_lost += 1
                        # Predict next position even if lost, but keep same points
                        prev_center = predicted_center 
                        tracked_points_list.append(tracked_points_list[-1])
                else:
                    frames_lost += 1
                    prev_center = predicted_center
                    tracked_points_list.append(tracked_points_list[-1])

                if opt.debug_tracking:
                    dbg_frame = frame_bgr.copy()
                    pts = tracked_points_list[-1]
                    cv2.circle(dbg_frame, tuple(map(int, pts[0])), 4, (255, 0, 0), -1)
                    cv2.circle(dbg_frame, tuple(map(int, pts[1])), 4, (0, 255, 0), -1)
                    if best_hull is not None:
                        cv2.drawContours(dbg_frame, [best_hull], -1, (0, 255, 255), 1)
                    cv2.putText(dbg_frame, f"Frame {i} Lost={frames_lost}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    debug_frames.append(dbg_frame)

        if not opt.use_pose and not opt.use_color:
            print("\n--- STEP 4: Tracking Cudgel in Driving Video (CSRT trackers)... ---")
            # Initialize trackers
            def get_tracker():
                try:
                    return cv2.TrackerCSRT_create()
                except AttributeError:
                    print("Error: TrackerCSRT_create not found. Try 'pip install opencv-contrib-python-headless'")
                    sys.exit(1)

            trackers = [get_tracker() for _ in range(2)]
            box_size = 16
            initial_boxes = []
            for pt in dst_pts:
                initial_boxes.append((int(pt[0] - box_size/2), int(pt[1] - box_size/2), box_size, box_size))
            
            for i in range(2):
                trackers[i].init(first_dr_frame, initial_boxes[i])

            if opt.debug_tracking and not debug_frames:
                debug_frames.append(first_dr_frame.copy())

            success_count = 0
            for i in tqdm(range(1, len(driving_video)), desc="CSRT Tracking"):
                frame_bgr = (driving_video[i] * 255).astype(np.uint8)
                current_pts = []
                all_ok = True
                for j in range(2):
                    ok, box = trackers[j].update(frame_bgr)
                    if ok:
                        cx, cy = box[0] + box[2]/2, box[1] + box[3]/2
                        current_pts.append([cx, cy])
                    else:
                        all_ok = False
                        break
                
                if all_ok:
                    tracked_points_list.append(current_pts)
                    success_count += 1
                else:
                    tracked_points_list.append(tracked_points_list[-1])

                if opt.debug_tracking:
                    dbg_frame = frame_bgr.copy()
                    pts = tracked_points_list[-1]
                    cv2.circle(dbg_frame, tuple(map(int, pts[0])), 4, (255, 0, 0), -1)
                    cv2.circle(dbg_frame, tuple(map(int, pts[1])), 4, (0, 255, 0), -1)
                    cv2.putText(dbg_frame, f"CSRT Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    debug_frames.append(dbg_frame)
    
    if opt.debug_tracking:
        print(f"Saving debug tracking video to {opt.debug_tracking}...")
        imageio.mimsave(opt.debug_tracking, debug_frames, fps=fps)
    
    if opt.debug_mask and 'debug_mask_frames' in locals() and debug_mask_frames:
        print(f"Saving debug mask video to {opt.debug_mask}...")
        imageio.mimsave(opt.debug_mask, debug_mask_frames, fps=fps)

    # Optional Smoothing
    if opt.smooth_points > 1:
        print(f"Smoothing tracked points (window size={opt.smooth_points})...")
        pts_arr = np.array(tracked_points_list) # (N, 2, 2)
        smoothed = np.empty_like(pts_arr)
        for p_idx in range(2):
            for c_idx in range(2):
                data = pts_arr[:, p_idx, c_idx]
                win = opt.smooth_points
                smoothed[:, p_idx, c_idx] = np.convolve(data, np.ones(win)/win, mode='same')
        tracked_points_list = smoothed.tolist()

    # Calculate fixed scale if requested
    initial_scale = None
    if opt.fixed_scale:
        dx_p = src_pts[1][0] - src_pts[0][0]
        dy_p = src_pts[1][1] - src_pts[0][1]
        dx_q = dst_pts[1][0] - dst_pts[0][0]
        dy_q = dst_pts[1][1] - dst_pts[0][1]
        initial_scale = np.hypot(dx_q, dy_q) / (np.hypot(dx_p, dy_p) + 1e-6)
        print(f"Fixed scale enabled. Scale set to {initial_scale:.4f}")

    # --- Step 5: Overlay Warped Cudgel onto Generated Frames ---
    print("\n--- STEP 5: Compositing Final Video... ---")
    final_frames = []
    
    for i in tqdm(range(len(predictions)), desc="Overlaying Cudgel"):
        pred_frame = predictions[i] * 255.0  # TPS output is [0, 1]
        
        # Calculate affine transform for this frame
        M = get_similarity_transform(src_pts, tracked_points_list[i], fixed_scale=initial_scale)
        
        # Warp the RGBA cudgel
        cudgel_warped_rgba = cv2.warpAffine(source_cudgel_rgba, M, (W, H), 
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        # Alpha Blending
        alpha_mask = cudgel_warped_rgba[:, :, 3] / 255.0
        alpha_mask = np.stack([alpha_mask]*3, axis=-1)
        
        cudgel_rgb = cudgel_warped_rgba[:, :, :3]
        
        # Combine
        final_frame = pred_frame * (1.0 - alpha_mask) + cudgel_rgb * alpha_mask
        final_frames.append(np.clip(final_frame, 0, 255).astype(np.uint8))

    # --- Step 6: Save Result ---
    print("\n--- Saving result video... ---")
    imageio.mimsave(opt.result_video, final_frames, fps=fps)
    print(f"Success! Saved to {opt.result_video}")
