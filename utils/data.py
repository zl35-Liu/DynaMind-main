import torch
from einops import rearrange
from typing import List
import decord


def split_video_into_clips(
        video_path: str,
        fps: int = 24,
        waste_time: int = 3,
        block_time: int = 13,
        clips_per_block: int = 5,
        num_blocks: int = 40,
        sample_rate: int = 8,
        final_frames_per_clip: int = 6,
        video_width: int = 512,
        video_height: int = 288,
) -> List[torch.Tensor]:
    """
    将单个视频文件按照指定的逻辑切割成多个小片段并进行帧采样。

    参数:
    - video_path (str): 视频文件的路径。
    - fps (int): 视频的帧率。
    - waste_time (int): 视频开头要跳过的秒数。
    - block_time (int): 每个大块的持续时间（秒）。
    - clips_per_block (int): 每个大块中包含的小片段数量。
    - num_blocks (int): 总共有多少个大块。
    - sample_rate (int): 帧采样率，例如 8 表示每 8 帧取一帧。
    - final_frames_per_clip (int): 每个小片段最终保留的帧数。
    - video_width (int): 视频帧的宽度。
    - video_height (int): 视频帧的高度。

    返回:
    - List[torch.Tensor]: 包含所有处理后视频片段的列表。
                         每个张量的形状为 (F, C, H, W)，例如 (6, 3, 288, 512)。
    """
    video_clips = []

    try:
        vr = decord.VideoReader(video_path, width=video_width, height=video_height)
    except Exception as e:
        print(f"无法加载视频 {video_path}: {e}")
        return video_clips

    total_frames = len(vr)

    for i in range(num_blocks):
        start_frame_of_block = int(waste_time * fps + block_time * fps * i)

        for j in range(clips_per_block):
            clip_frame_length = int(2 * fps)  # 2秒的片段
            start_frame_of_clip = start_frame_of_block + clip_frame_length * j + 1
            end_frame_of_clip = start_frame_of_clip + clip_frame_length

            if start_frame_of_clip >= total_frames:
                continue
            if end_frame_of_clip > total_frames:
                end_frame_of_clip = total_frames

            # 获取帧
            clip = vr.get_batch(range(start_frame_of_clip, end_frame_of_clip))

            # 将 decord 的张量转换为 PyTorch 张量
            clip = torch.from_numpy(clip.asnumpy())

            # 转换为 (frames, channels, height, width)
            clip = rearrange(clip, "f h w c -> f c h w")

            # 帧采样
            video = clip[::sample_rate][:final_frames_per_clip]

            if video.shape[0] == final_frames_per_clip:
                video_clips.append(video)

    return video_clips

