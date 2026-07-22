import numpy as np
import torch
from einops import rearrange
from typing import List
import decord
import cv2


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
    Split one video into clips and sample frames according to the configured frame rate, initial skip time, block duration, clips per block, frames per clip, sampling rate, and output resolution.
    """
    video_clips = []

    try:
        vr = decord.VideoReader(video_path, width=video_width, height=video_height)
    except Exception as e:
        print(f"Failed to load video {video_path}: {e}")
        return video_clips

    total_frames = len(vr)

    for i in range(num_blocks):
        start_frame_of_block = int(waste_time * fps + block_time * fps * i)

        for j in range(clips_per_block):
            clip_frame_length = int(2 * fps)
            start_frame_of_clip = start_frame_of_block + clip_frame_length * j + 1
            end_frame_of_clip = start_frame_of_clip + clip_frame_length

            if start_frame_of_clip >= total_frames:
                continue
            if end_frame_of_clip > total_frames:
                end_frame_of_clip = total_frames


            clip = vr.get_batch(range(start_frame_of_clip, end_frame_of_clip))


            clip = torch.from_numpy(clip.asnumpy())


            clip = rearrange(clip, "f h w c -> f c h w")


            video = clip[::sample_rate][:final_frames_per_clip]

            if video.shape[0] == final_frames_per_clip:
                video_clips.append(video)

    return video_clips


def split_cine_into_clips(
    video_path: str,
    sample_rate: int = 3,
    final_frames_per_clip: int = 16,
    resize_h: int = 288,
    resize_w: int = 512,
):
    """
    Sample frames from a video and divide them into fixed-length clips with optional resizing. Return the clip tensor together with the source frames per second.
    """

    try:

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        if total_frames < 2:
            print(f"⚠️ Skipping short video {video_path}, too few frames.")
            return []


        frame_indices = np.arange(0, total_frames, sample_rate)
        frame_indices = frame_indices[frame_indices < total_frames]


        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]


        frames_resized = [
            cv2.resize(f, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            for f in frames
        ]


        frames_tensor = torch.from_numpy(np.stack(frames_resized)).permute(0, 3, 1, 2)


        clips = []
        num_frames = frames_tensor.shape[0]
        stride = final_frames_per_clip

        for start in range(0, num_frames - final_frames_per_clip + 1, stride):
            clip = frames_tensor[start : start + final_frames_per_clip]
            clips.append(clip)


        if len(clips) == 0 and num_frames >= final_frames_per_clip // 2:
            pad_frames = final_frames_per_clip - num_frames
            pad = frames_tensor[-1:].repeat(pad_frames, 1, 1, 1)
            clip = torch.cat((frames_tensor, pad), dim=0)
            clips.append(clip)

        return clips

    except Exception as e:
        print(f"❌ split_cine_into_clips failed: {video_path} -> {e}")
        return []