import imageio
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import color
import os
import matplotlib.pyplot as plt
import cv2


def extract_gif_frames(gif_path):
    """
    Extract every frame from a GIF as a list of images.
    """
    gif = imageio.mimread(gif_path)

    return gif


def compute_video_ssim(video1, video2, use_grayscale=False,use_rgba=False, dynamic_range=None):
    """
    Compute SSIM between two video arrays shaped (T, C, H, W). Optionally convert frames to grayscale and infer the dynamic range automatically.
    """

    if video1.shape != video2.shape:
        raise ValueError(f"Video shapes do not match: {video1.shape} vs {video2.shape}")

    T, C, H, W = video1.shape


    if dynamic_range is None:
        if np.issubdtype(video1.dtype, np.integer):
            dynamic_range = np.iinfo(video1.dtype).max
        else:
            dynamic_range = 1.0

    frame_ssims = []

    for t in range(T):
        frame1 = video1[t]
        frame2 = video2[t]


        if use_rgba:

            frame1_rgb = np.moveaxis(frame1, 0, -1)
            frame2_rgb = np.moveaxis(frame2, 0, -1)


            frame1 = color.rgba2rgb(frame1_rgb)
            frame2 = color.rgba2rgb(frame2_rgb)


            frame1 = np.moveaxis(frame1, -1, 0)
            frame2 = np.moveaxis(frame2, -1, 0)
            C = 3


        if use_grayscale:
            if C > 1:

                gray1 = color.rgb2gray(np.moveaxis(frame1, 0, -1))
                gray2 = color.rgb2gray(np.moveaxis(frame2, 0, -1))
            else:

                gray1 = frame1[0]
                gray2 = frame2[0]


            ssim_val = ssim(
                gray1, gray2,
                data_range=dynamic_range,
                win_size=3,
                gaussian_weights=True
            )


        else:

            frame1_rgb = np.moveaxis(frame1, 0, -1)
            frame2_rgb = np.moveaxis(frame2, 0, -1)


            ssim_val = ssim(
                frame1_rgb, frame2_rgb,
                data_range=dynamic_range,
                multichannel=True,
                win_size=3,
                gaussian_weights=True,
                channel_axis=-1
            )

        frame_ssims.append(ssim_val)

    return np.mean(frame_ssims), frame_ssims


def compute_frame_ssim(frame1, frame2, use_grayscale=True, use_rgba=False,dynamic_range=None):
    """
    Compute SSIM between two individual frames shaped (H, W, C), optionally in grayscale and with a configurable dynamic range.
    """
    H, W, C = frame1.shape


    if dynamic_range is None:
        if np.issubdtype(frame1.dtype, np.integer):
            dynamic_range = np.iinfo(frame1.dtype).max
        else:
            dynamic_range = 1.0


    if use_rgba:
        frame1 = color.rgba2rgb(frame1)
        frame2 = color.rgba2rgb(frame2)
        C = 3


    if use_grayscale:
        if C > 1:
            gray1 = color.rgb2gray(frame1)
            gray2 = color.rgb2gray(frame2)
        else:
            gray1 = frame1[..., 0]
            gray2 = frame2[..., 0]

        return ssim(
            gray1, gray2,
            data_range=dynamic_range,
            win_size=3,
            gaussian_weights=True
        )


    return ssim(
        frame1, frame2,
        data_range=dynamic_range,
        multichannel=True,
        win_size=3,
        gaussian_weights=True,
        channel_axis=-1
    )


def compare_gif_sequences(video_gt, video_gen, degradation_method =['jpeg_compression','downsample_upsample','gaussian_blur'], use_grayscale=False, use_rgba=False):
    """
    Compare SSIM for two groups of video sequences shaped (videos, frames, height, width, channels) and return per-video and aggregate results.
    """

    if video_gt.shape != video_gen.shape:
        raise ValueError(f"Video array shapes do not match: {video_gt.shape} vs {video_gen.shape}")

    nums, frames, H, W, C = video_gt.shape


    dynamic_range = None
    if np.issubdtype(video_gt.dtype, np.integer):
        dynamic_range = np.iinfo(video_gt.dtype).max



    results = {}


    for i in range(nums):
        video1 = video_gt[i]
        video2 = video_gen[i]

        frame_ssims = []


        for j in range(frames):
            frame1 = video1[j]
            frame2 = video2[j]


            if frame1.ndim == 3 and frame1.shape[2] == C:
                frame1 = frame1.transpose(1, 2, 0)
                frame2 = frame2.transpose(1, 2, 0)
                pass
            elif frame1.ndim == 4 and frame1.shape[1] == C:
                frame1 = frame1.transpose(1, 2, 0)
                frame2 = frame2.transpose(1, 2, 0)
            else:
                raise ValueError(f"Invalid frame shape: {frame1.shape}")













            if 'gaussian_blur' in degradation_method:

                blur_sigma = 3
                frame1 = cv2.GaussianBlur(frame1, (0, 0), blur_sigma)







            print("Compressed output shape",frame1.shape,frame2.shape)


            ssim_val = compute_frame_ssim(
                frame1, frame2,
                use_grayscale=use_grayscale,
                use_rgba = use_rgba,
                dynamic_range=dynamic_range
            )
            frame_ssims.append(ssim_val)


        results[f'video_{i}'] = {
            'frame_ssims': frame_ssims,
            'mean_ssim': np.mean(frame_ssims),
            'min_ssim': np.min(frame_ssims),
            'max_ssim': np.max(frame_ssims),
            'std_ssim': np.std(frame_ssims)
        }

    return results


def load_frames_in_path(path):
    frames = np.empty((0, 6,3,288,512))
    gif_files = sorted([f for f in os.listdir(path) if f.lower().endswith('.gif')])
    length = len(gif_files)
    for f in gif_files:
        gif_path = os.path.join(path, f)


        gif = extract_gif_frames(gif_path)
        gif_frames = np.empty((0, 3,288,512))
        for frame in gif:
            frame = frame.transpose(2, 0, 1)
            gif_frames = np.concatenate((gif_frames, frame[None, ...]), axis=0)
        frames = np.concatenate((frames, gif_frames[None, ...]), axis=0)

    print(frames.shape)
    np.save("/path/to/workspace/Ljx/EEG2Video-main/acessment/video_frames_gen.npy", frames)

    return frames


def visualize_comparison(results):
    """
    Visualize the results returned by compare_gif_sequences.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)


    for video_id, data in results.items():
        ax1.plot(data['frame_ssims'], label=f'{video_id} (avg: {data["mean_ssim"]:.4f})')

    ax1.set_title('Per-Frame SSIM Comparison Across Videos')
    ax1.set_ylabel('SSIM Value')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax1.grid(True)


    video_ids = list(results.keys())
    mean_ssims = [results[v]['mean_ssim'] for v in video_ids]
    min_ssims = [results[v]['min_ssim'] for v in video_ids]
    max_ssims = [results[v]['max_ssim'] for v in video_ids]

    bar_width = 0.25
    index = np.arange(len(video_ids))

    ax2.bar(index, mean_ssims, bar_width, label='Mean SSIM')
    ax2.bar(index + bar_width, min_ssims, bar_width, label='Minimum SSIM', alpha=0.7)
    ax2.bar(index + 2 * bar_width, max_ssims, bar_width, label='Maximum SSIM', alpha=0.7)

    ax2.set_title('SSIM Comparison Across Videos')
    ax2.set_xlabel('Video ID')
    ax2.set_ylabel('SSIM Value')
    ax2.set_xticks(index + bar_width)
    ax2.set_xticklabels(video_ids)
    ax2.legend()
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":





    sub = "1_session2"
    block_num = 0
    gif_sub_name = f'sub{sub}/{block_num}'
    dir1 = f"/path/to/workspace/Disk/hdd-1/Ljx/EEG2Video-main/EEG2Video/reconstruction/40/{gif_sub_name}"
    video_gt = np.load("/path/to/workspace/Ljx/EEG2Video-main/acessment/video_frames.npy")
    video_gt = video_gt[block_num*200:(block_num+1)*200]
    print(video_gt.shape)

    video_gen = load_frames_in_path(dir1)

    print("Checking data types and ranges for both video groups")
    video_gen = video_gen.astype(np.float32)/255.0
    print(video_gen.dtype,video_gen.max(),video_gen.min())
    video_gt = video_gt.astype(np.float32)/255.0
    print(video_gt.dtype,video_gt.max(),video_gt.min())

    comparison_results = compare_gif_sequences(video_gt, video_gen)

    ssim_list = []

    print("\nSSIM comparison results:")
    print("=" * 50)
    for gif_name, data in comparison_results.items():
        print(f"GIF: {gif_name}")
        print(f"  Mean SSIM: {data['mean_ssim']:.4f}")
        print(f"  Minimum SSIM: {data['min_ssim']:.4f} (frame  {np.argmin(data['frame_ssims']) + 1})")
        print(f"  Maximum SSIM: {data['max_ssim']:.4f} (frame  {np.argmax(data['frame_ssims']) + 1})")
        ssim_list.append(data['max_ssim'])
        print("-" * 50)


    visualize_comparison(comparison_results)
    print(ssim_list)
    print(f"{len(ssim_list)} image pairs; mean SSIM  {sum(ssim_list)/len(ssim_list)}")
