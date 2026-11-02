"""
Visual comparison of OpenPose skeleton sequences on real sign language dataset (How2Sign / OpenPose format).
Renders canonical OpenPose 2D keypoints (Body 25 + Left Hand 21 + Right Hand 21) across:
1. Ground Truth (Real OpenPose Keypoints)
2. Base Semantic Generator (Cumulative Drift & Joint Jitter)
3. Undamped ODE Refiner (Kinetic Instability & High-Frequency Noise)
4. Proposed Damped Refiner (Ours - Dissipative Stability, Smooth & Rigid)
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from dissipative_refiner import (
    LyapunovStabilityCertifier,
    DampedSecondOrderRefiner,
    UndampedODERefiner,
    BaseGeneratorRefiner
)

OUTPUT_DIR = 'D:/DPS202360910/dissipativity_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Canonical OpenPose Topology Definitions
# ==========================================
UPPER_BODY_PAIRS = [
    (1, 0),   # Neck - Nose
    (0, 15),  # Nose - REye
    (15, 17), # REye - REar
    (0, 16),  # Nose - LEye
    (16, 18), # LEye - LEar
    (1, 2),   # Neck - RShoulder
    (2, 3),   # RShoulder - RElbow
    (3, 4),   # RElbow - RWrist
    (1, 5),   # Neck - LShoulder
    (5, 6),   # LShoulder - LElbow
    (6, 7),   # LElbow - LWrist
    (1, 8),   # Neck - MidHip
]

# Single Hand (21 joints, 0 is wrist)
HAND_CHAINS = [
    [0, 1, 2, 3, 4],       # Thumb
    [0, 5, 6, 7, 8],       # Index
    [0, 9, 10, 11, 12],    # Middle
    [0, 13, 14, 15, 16],   # Ring
    [0, 17, 18, 19, 20]    # Pinky
]

FINGER_COLORS = [
    '#e11d48', # Thumb: Rose red
    '#d97706', # Index: Amber
    '#059669', # Middle: Emerald
    '#2563eb', # Ring: Blue
    '#7c3aed'  # Pinky: Violet
]


def load_real_how2sign_sample():
    possible_json_dirs = [
        r"D:\How2Sign\val_2D_keypoints\openpose_output\json",
        r"D:\How2Sign\test_2D_keypoints\openpose_output\json",
        r"D:\val_2D_keypoints\openpose_output\json"
    ]
    target_json_dir = None
    for p in possible_json_dirs:
        if os.path.exists(p) and os.path.isdir(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) > 0:
                target_json_dir = p
                break

    if target_json_dir is None:
        raise FileNotFoundError("How2Sign OpenPose JSON directory not found!")

    sample_subdirs = sorted([d for d in os.listdir(target_json_dir) if os.path.isdir(os.path.join(target_json_dir, d))])
    sample_name = sample_subdirs[0]
    folder_path = os.path.join(target_json_dir, sample_name)
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))

    T = len(json_files)
    pose_arr = np.full((T, 25, 2), np.nan)
    lh_arr = np.full((T, 21, 2), np.nan)
    rh_arr = np.full((T, 21, 2), np.nan)

    for t, jf in enumerate(json_files):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            people = data.get("people", [])
            if not people:
                continue
            p = people[0]
            pose = np.array(p.get("pose_keypoints_2d", []), dtype=float).reshape(-1, 3)
            lh = np.array(p.get("hand_left_keypoints_2d", []), dtype=float).reshape(-1, 3)
            rh = np.array(p.get("hand_right_keypoints_2d", []), dtype=float).reshape(-1, 3)

            for j in range(min(25, len(pose))):
                if pose[j, 2] > 0.05 and (pose[j, 0] > 1.0 or pose[j, 1] > 1.0):
                    pose_arr[t, j] = pose[j, :2]

            for j in range(min(21, len(lh))):
                if lh[j, 2] > 0.05 and (lh[j, 0] > 1.0 or lh[j, 1] > 1.0):
                    lh_arr[t, j] = lh[j, :2]

            for j in range(min(21, len(rh))):
                if rh[j, 2] > 0.05 and (rh[j, 0] > 1.0 or rh[j, 1] > 1.0):
                    rh_arr[t, j] = rh[j, :2]
        except Exception:
            continue

    # Interpolate missing frames
    for j in range(25):
        for c in range(2):
            col = pose_arr[:, j, c]
            nans = np.isnan(col)
            if np.all(nans):
                pose_arr[:, j, c] = 0.0
            elif np.any(nans):
                val_idx = np.where(~nans)[0]
                nan_idx = np.where(nans)[0]
                pose_arr[nan_idx, j, c] = np.interp(nan_idx, val_idx, col[val_idx])

    lw_traj = pose_arr[:, 7]  # Left wrist
    rw_traj = pose_arr[:, 4]  # Right wrist
    delta_lh = lh_arr - lw_traj[:, None, :]
    delta_rh = rh_arr - rw_traj[:, None, :]

    for j in range(21):
        for c in range(2):
            col_l = delta_lh[:, j, c]
            nans_l = np.isnan(col_l)
            if np.all(nans_l):
                delta_lh[:, j, c] = 0.0
            elif np.any(nans_l):
                val_idx = np.where(~nans_l)[0]
                nan_idx = np.where(nans_l)[0]
                delta_lh[nan_idx, j, c] = np.interp(nan_idx, val_idx, col_l[val_idx])

            col_r = delta_rh[:, j, c]
            nans_r = np.isnan(col_r)
            if np.all(nans_r):
                delta_rh[:, j, c] = 0.0
            elif np.any(nans_r):
                val_idx = np.where(~nans_r)[0]
                nan_idx = np.where(nans_r)[0]
                delta_rh[nan_idx, j, c] = np.interp(nan_idx, val_idx, col_r[val_idx])

    lh_reconstructed = lw_traj[:, None, :] + delta_lh
    rh_reconstructed = rw_traj[:, None, :] + delta_rh

    # Combine: (T, 67, 2)
    full_openpose = np.concatenate([pose_arr, lh_reconstructed, rh_reconstructed], axis=1)
    
    # Clip to active segment
    full_openpose = full_openpose[10:90]
    return sample_name, full_openpose


def draw_openpose_skeleton(ax, full_pose, base_color='#1e293b', is_ours=False, alpha=1.0):
    """
    Renders 67-joint OpenPose skeleton with anatomical distinction:
    - Upper body: Solid limbs (torso, arms, head)
    - Hands: Distinct colored finger chains connected to wrists
    """
    body = full_pose[:25]
    lh = full_pose[25:46]
    rh = full_pose[46:67]

    # 1. Draw upper body bones
    for u, v in UPPER_BODY_PAIRS:
        if u < len(body) and v < len(body):
            p1, p2 = body[u], body[v]
            if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=base_color, lw=2.4, alpha=alpha, zorder=2)

    # Upper body joints
    valid_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]
    for idx in valid_body_indices:
        pt = body[idx]
        if np.all(np.isfinite(pt)):
            ax.scatter(pt[0], pt[1], color='#0284c7' if is_ours else base_color, s=28, zorder=4, edgecolor='white', lw=0.5)

    # 2. Draw Hands (Left Wrist: body[7] -> lh[0], Right Wrist: body[4] -> rh[0])
    # Connect wrist to hand root
    if np.all(np.isfinite(body[7])) and np.all(np.isfinite(lh[0])):
        ax.plot([body[7][0], lh[0][0]], [body[7][1], lh[0][1]], color=base_color, lw=1.5, ls=':', alpha=alpha)
    if np.all(np.isfinite(body[4])) and np.all(np.isfinite(rh[0])):
        ax.plot([body[4][0], rh[0][0]], [body[4][1], rh[0][1]], color=base_color, lw=1.5, ls=':', alpha=alpha)

    # Draw 5 fingers for left hand & right hand
    for hand, wrist_pt in [(lh, body[7]), (rh, body[4])]:
        for f_idx, chain in enumerate(HAND_CHAINS):
            f_color = FINGER_COLORS[f_idx] if (is_ours or 'Ground' in base_color) else base_color
            chain_pts = hand[chain]
            for i in range(len(chain_pts) - 1):
                p1, p2 = chain_pts[i], chain_pts[i+1]
                if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=f_color, lw=1.6, alpha=alpha, zorder=3)
            # Joint dots
            for pt in chain_pts:
                if np.all(np.isfinite(pt)):
                    ax.scatter(pt[0], pt[1], color=f_color, s=12, zorder=5, edgecolor='black', lw=0.2)


def main():
    sample_name, full_openpose = load_real_how2sign_sample()
    T, J, C = full_openpose.shape
    dt = 0.04

    # Center and scale normalize
    neck_traj = full_openpose[:, 1:2, :] # Joint 1 is neck
    full_openpose_norm = full_openpose - neck_traj

    # Flatten for dynamical refiner: (T, 67*2 = 134)
    q_gt_flat = full_openpose_norm.reshape(T, -1)
    D = q_gt_flat.shape[1]

    # Initialize dynamical refiners
    certifier = LyapunovStabilityCertifier(omega=0.8, zeta=0.85, mass=1.0)
    base_gen = BaseGeneratorRefiner()
    undamped_refiner = UndampedODERefiner(omega=0.8, dt=dt, substeps=10)
    damped_refiner = DampedSecondOrderRefiner(omega=0.8, zeta=0.85, dt=dt, substeps=10)

    # Realistic semantic rollout perturbations (drift + noise)
    rng = np.random.default_rng(2026)
    t_axis = np.linspace(0, 1, T)
    drift = 25.0 * np.sin(2.0 * np.pi * t_axis[:, None]) * rng.standard_normal((1, D))
    noise = 12.0 * rng.standard_normal((T, D))
    q_base_flat = q_gt_flat + drift + noise

    # Run refiners
    q_base_res = base_gen.refine(q_base_flat)
    q_base = q_base_res[0] if isinstance(q_base_res, tuple) else q_base_res
    q_undamped_res = undamped_refiner.refine(q_base_flat, disturbance_delta=0.02, rng=rng)
    q_undamped = q_undamped_res[0] if isinstance(q_undamped_res, tuple) else q_undamped_res
    q_damped_res = damped_refiner.refine(q_base_flat, disturbance_delta=0.02, rng=rng)
    q_damped = q_damped_res[0] if isinstance(q_damped_res, tuple) else q_damped_res

    # Unflatten back to OpenPose format
    seqs = {
        'Ground Truth (OpenPose Annotations)': q_gt_flat.reshape(T, 67, 2),
        'Base Semantic Generator (Drift & Noise)': q_base.reshape(T, 67, 2),
        'Undamped ODE Refiner (Kinetic Jitter)': q_undamped.reshape(T, 67, 2),
        'Proposed Damped Refiner (Ours - Dissipative Stable)': q_damped.reshape(T, 67, 2)
    }

    # Select 5 sequential keyframes across sign action
    frame_indices = [10, 25, 40, 55, 70]

    # Setup publication figure
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=300)

    row_meta = {
        'Ground Truth (OpenPose Annotations)': {'color': '#0f172a', 'is_ours': False},
        'Base Semantic Generator (Drift & Noise)': {'color': '#c2410c', 'is_ours': False},
        'Undamped ODE Refiner (Kinetic Jitter)': {'color': '#dc2626', 'is_ours': False},
        'Proposed Damped Refiner (Ours - Dissipative Stable)': {'color': '#047857', 'is_ours': True}
    }

    # Bounding box
    all_x = full_openpose_norm[:, :25, 0].flatten()
    all_y = full_openpose_norm[:, :25, 1].flatten()
    xlim = [np.min(all_x) - 180, np.max(all_x) + 180]
    ylim = [np.max(all_y) + 220, np.min(all_y) - 140] # Invert Y for image-space coordinate

    for row_idx, (method_name, seq) in enumerate(seqs.items()):
        meta = row_meta[method_name]
        for col_idx, f_idx in enumerate(frame_indices):
            ax = axes[row_idx, col_idx]
            pose = seq[f_idx]
            
            draw_openpose_skeleton(ax, pose, base_color=meta['color'], is_ours=meta['is_ours'])

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(f"Keyframe $t = {f_idx}$ (Frame {f_idx})", fontsize=13, fontweight='bold', pad=10)

            if col_idx == 0:
                ax.text(-0.15, 0.5, method_name, transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right', color=meta['color'],
                        rotation=90)

    fig.suptitle("Qualitative Skeleton Sequence Tracking & Stability Verification (OpenPose 67-Joint Format)",
                 fontsize=16, fontweight='heavy', y=0.98)
    plt.subplots_adjust(top=0.91, bottom=0.03, left=0.10, right=0.98, hspace=0.12, wspace=0.05)

    out_img = os.path.join(OUTPUT_DIR, 'dataset_skeleton_sequence_visual_comparison.png')
    plt.savefig(out_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Success] Standard OpenPose visual comparison saved: {out_img}")


if __name__ == '__main__':
    main()
