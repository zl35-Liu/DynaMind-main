import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



GT_VIDEO_ROOT_DIR = '/path/to/cinebrain/dataset/clips'


RECON_VIDEO_DIR = '/path/to/DynaMind-main/outputs/reconstruction/cine/0-4'





GT_INDICES_TO_USE = list(range(0, 200))




def load_video_frames(video_path):
    """
    Load a video with OpenCV and return its frames in RGB order.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():

        print(f"Warning: Could not open video file: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames

def calculate_frame_metrics(gt_frame, recon_frame):
    """
    Compute SSIM and PSNR for one frame pair. Both metrics use the configured pixel-value data range.
    """

    try:


        data_range = gt_frame.max() - gt_frame.min()
        if data_range == 0:

             return 0.0, 0.0

        ssim_val = ssim(gt_frame, recon_frame, channel_axis=2, data_range=data_range, multichannel=True)


        psnr_val = psnr(gt_frame, recon_frame, data_range=data_range)

        return ssim_val, psnr_val
    except Exception as e:
        print(f"Error calculating metrics: {e}")

        return 0.0, 0.0



def run_evaluation(gt_root_dir, recon_dir, indices_to_use):
    """
    Run the main SSIM and PSNR evaluation procedure.
    """
    all_ssim_values = []
    all_psnr_values = []
    processed_count = 0

    print(f"Starting evaluation for {len(indices_to_use)} video indices.")



    gt_paths_map = {}
    for index in indices_to_use:

        video_id = f"{index:06d}"
        gt_path = os.path.join(gt_root_dir, f"{video_id}.mp4")
        if os.path.exists(gt_path):
            gt_paths_map[video_id] = gt_path
        else:
            print(f"Warning: GT video {gt_path} not found.")


    for video_id, gt_path in gt_paths_map.items():
        video_id = int(video_id)

        recon_path = os.path.join(recon_dir, f"{video_id}.mp4")

        if not os.path.exists(recon_path):
            print(f"Warning: Reconstructed video for {video_id} not found. Skipping.")
            continue


        gt_frames = load_video_frames(gt_path)
        recon_frames = load_video_frames(recon_path)

        if not gt_frames or not recon_frames:
            continue

        gt_frames = gt_frames[:len(recon_frames)]

        if len(gt_frames) != len(recon_frames):
            print(f"Warning: Frames count mismatch for {video_id}. GT: {len(gt_frames)}, Recon: {len(recon_frames)}. Skipping.")
            continue


        for gt_frame, recon_frame in zip(gt_frames, recon_frames):

            if gt_frame.shape != recon_frame.shape:
                print(f"Warning: Frame shape mismatch for {video_id}. GT: {gt_frame.shape}, Recon: {recon_frame.shape}. Skipping frame.")
                continue

            ssim_val, psnr_val = calculate_frame_metrics(gt_frame, recon_frame)
            print(f"Video ID {video_id}, Frame {processed_count + 1}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}")


            all_ssim_values.append(ssim_val)
            all_psnr_values.append(psnr_val)


        processed_count += 1




    if not all_ssim_values:
        return {"SSIM_mean": 0.0, "PSNR_mean": 0.0, "Total_Frames": 0}

    total_frames = len(all_ssim_values)
    mean_ssim = np.mean(all_ssim_values)
    mean_psnr = np.mean(all_psnr_values)

    return {
        "SSIM_mean": mean_ssim,
        "PSNR_mean": mean_psnr,
        "Total_Frames": total_frames
    }

if __name__ == "__main__":


    if GT_VIDEO_ROOT_DIR == '/path/to/cinebrain_gt_videos' or \
       RECON_VIDEO_DIR == '/path/to/reconstructed_videos':
        print("--- Error: configure GT_VIDEO_ROOT_DIR and RECON_VIDEO_DIR at the top of the script first. ---")
    else:
        results = run_evaluation(GT_VIDEO_ROOT_DIR, RECON_VIDEO_DIR, GT_INDICES_TO_USE)

        print("\n--- Video Reconstruction Evaluation Results (SSIM and PSNR) ---")
        print(f"Total analyzed frames: {results['Total_Frames']}")
        print(f"Mean structural similarity (SSIM): {results['SSIM_mean']:.4f}")
        print(f"Mean peak signal-to-noise ratio (PSNR): {results['PSNR_mean']:.2f}")