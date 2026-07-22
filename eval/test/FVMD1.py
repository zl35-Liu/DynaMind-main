import os
import numpy as np
from fvmd import fvmd
import shutil
from PIL import Image
import imageio
from acessment.FVMD import load_and_preprocess_video


TARGET_NUM_FRAMES = 6
TARGET_HEIGHT = 288
TARGET_WIDTH = 512
TARGET_CHANNELS = 3
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)



def save_frames_as_images(video_frames: np.ndarray, output_dir: str, filename_prefix: str = "Frame"):
    """
    Save a NumPy video array shaped (frames, height, width, channels) as an image sequence in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("ok")
    video_frames = video_frames.astype(np.uint8)
    for i, frame in enumerate(video_frames):
        img_path = os.path.join(output_dir, f"{filename_prefix}{i + 1}.png")
        img = Image.fromarray(frame)
        img.save(img_path)




def prepare_fvmd_data_images(
        gen_video_dir: str,
        gt_video_dir: str,
        output_base_dir: str = "fvmd_data_images"
) -> tuple[str, str]:
    """
    Convert generated and ground-truth videos into image sequences organized in the directory structure expected by FVMD. Return both prepared root directories.
    """
    if not os.path.isdir(gen_video_dir) or not os.path.isdir(gt_video_dir):
        raise ValueError("Input paths must be valid directories.")

    gen_files = sorted([f for f in os.listdir(gen_video_dir) if f.split('.')[0].isdigit()],
                       key=lambda x: int(x.split('.')[0]))
    gt_files = sorted([f for f in os.listdir(gt_video_dir) if f.split('.')[0].isdigit()],
                      key=lambda x: int(x.split('.')[0]))

    num_pairs = min(len(gen_files), len(gt_files))
    if num_pairs == 0:
        raise ValueError("No matching video files were found in the directories.")

    gen_output_dir = os.path.join(output_base_dir, "gen_ready")
    gt_output_dir = os.path.join(output_base_dir, "gt_ready")


    if os.path.exists(output_base_dir): shutil.rmtree(output_base_dir)
    os.makedirs(gen_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    print(f"Loading and preparing  {num_pairs}  video pairs...")

    for i in range(num_pairs):
        gen_path = os.path.join(gen_video_dir, gen_files[i])
        gt_path = os.path.join(gt_video_dir, gt_files[i])

        gen_video_frames = load_and_preprocess_video(gen_path)
        gt_video_frames = load_and_preprocess_video(gt_path)

        if gen_video_frames is not None and gt_video_frames is not None:
            gen_clip_dir = os.path.join(gen_output_dir, f"Clip{i}")
            gt_clip_dir = os.path.join(gt_output_dir, f"Clip{i}")

            save_frames_as_images(gen_video_frames, gen_clip_dir)
            save_frames_as_images(gt_video_frames, gt_clip_dir)
        else:
            print(f"Warning: skipping video pair  {gen_files[i]} vs {gt_files[i]}。")

    print("Data directories are ready.")
    return gen_output_dir, gt_output_dir



def calculate_overall_fvmd_from_images(
        gen_ready_dir: str,
        gt_ready_dir: str,
        log_base_dir: str = "fvmd_logs"
) -> dict:
    """
    Calculate the overall FVMD score from prepared generated and ground-truth image-sequence directories and return the metric results.
    """
    if not os.path.isdir(gen_ready_dir) or not os.path.isdir(gt_ready_dir):
        raise ValueError("The prepared data directories do not exist.")

    run_log_dir = os.path.join(log_base_dir, f"fvmd_run_{os.getpid()}")
    os.makedirs(run_log_dir, exist_ok=True)

    print(f"Starting overall FVMD calculation...")


    fvmd_score = fvmd(
        gen_path=gen_ready_dir,
        gt_path=gt_ready_dir,
        log_dir=run_log_dir
    ).item()

    results = {'overall_fvmd_score': fvmd_score}
    print(f"\nOverall FVMD score: {fvmd_score:.4f}")






    return results



if __name__ == "__main__":
    sub = 1
    block_num = 0
    gif_sub_name = f'sub{sub}/{block_num}'
    GENERATED_GIF_DIR = f"/path/to/workspace/Disk/hdd-1/Ljx/EEG2Video-main/EEG2Video/reconstruction/40/{gif_sub_name}"
    GROUND_TRUTH_MP4_DIR = f"/path/to/workspace/Ljx/EEG2Video-main/acessment/video_gt/{block_num}"

    GENERATED_GIF_DIR = GROUND_TRUTH_MP4_DIR
    output_base_dir = f"/path/to/workspace/Ljx/EEG2Video-main/acessment/fvmd_results/{gif_sub_name}"

    try:

        gen_ready_dir, gt_ready_dir = prepare_fvmd_data_images(GENERATED_GIF_DIR, GROUND_TRUTH_MP4_DIR, output_base_dir)


        results = calculate_overall_fvmd_from_images(gen_ready_dir, gt_ready_dir)


        print("\nFinal FVMD results:", results)
    except Exception as e:
        print(f"A fatal error occurred: {e}")
