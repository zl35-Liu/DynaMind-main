import os
import decord
import torch
from transformers import CLIPModel, CLIPImageProcessor
from einops import rearrange
from typing import List, Optional
import cv2
import numpy as np


# ==============================================================================
# 视频处理函数：切分、采样帧并返回张量列表
# ==============================================================================
def process_videos_to_tensors(
        video_dir: str,
        sub_dir: str = "video_tensor",
        output_height: int = 288,
        output_width: int = 512
) -> torch.Tensor:
    """
    处理指定目录下的7个MP4视频，切分、采样帧并保存为短视频片段，同时返回处理后的张量列表。

    Args:
        video_dir (str): 包含7个MP4视频文件的目录路径。
        output_clips_dir (str): 保存短视频片段的子目录路径。
        output_height (int): 视频帧的输出高度。
        output_width (int): 视频帧的输出宽度。

    Returns:
        List[torch.Tensor]: 包含1400个视频片段的张量列表，每个张量形状为 (6, 3, H, W)。
    """
    output_clips_dir = os.path.join(video_data_dir, "video_clips")
    output_tensors_dir = os.path.join(video_data_dir, "video_tensor")
    if os.path.exists(output_tensors_dir):
        filename = os.listdir(output_tensors_dir)[0]
        processed_video = torch.load(os.path.join(output_tensors_dir, filename))
        return processed_video

    if not os.path.exists(output_clips_dir):
        os.makedirs(output_clips_dir)
        os.makedirs(output_tensors_dir)
        print(f"Created output directory: {output_clips_dir}")

    processed_video_list: List[torch.Tensor] = []

    # 按照提供的逻辑设置参数
    fps = 24
    waste_time = 3
    block_time = 13

    video_counter = 0

    for k in range(1, 8):
        path = os.path.join(video_dir, f"{k}.mp4")
        if not os.path.exists(path):
            print(f"Warning: Video file not found at {path}. Skipping.")
            continue

        print(f"Processing video {k}.mp4...")

        try:
            vr = decord.VideoReader(path, width=output_width, height=output_height)
        except decord.DecordError as e:
            print(f"Error reading video file {path}: {e}. Skipping.")
            continue

        total_frames = len(vr)
        max_frame_index = total_frames - 1

        for i in range(40):
            start_block_frame = int(waste_time * fps + block_time * fps * i)

            for j in range(5):
                clip_frame_length = int(2 * fps)
                start_frame = start_block_frame + clip_frame_length * j + 1
                end_frame = start_frame + clip_frame_length

                if start_frame > max_frame_index:
                    break

                end_frame = min(end_frame, total_frames)

                clip_frames = vr.get_batch([i for i in range(start_frame, end_frame)]).asnumpy()
                sampled_frames = clip_frames[::8]

                if len(sampled_frames) != 6:
                    print(f"Warning: Skipped a clip from video {k}.mp4. Expected 6 frames, got {len(sampled_frames)}.")
                    continue

                clip_tensor_np = rearrange(sampled_frames, "f h w c -> f c h w")
                processed_video_list.append(torch.from_numpy(clip_tensor_np))

                # 保存为MP4文件
                video_counter += 1
                output_clip_path = os.path.join(output_clips_dir, f"{video_counter}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_clip_path, fourcc, 24.0, (output_width, output_height))

                for frame_tensor_np in clip_tensor_np:
                    frame = rearrange(frame_tensor_np, "c h w -> h w c")
                    frame = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_RGB2BGR)
                    out.write(frame)
                out.release()

    processed_video = torch.stack(processed_video_list)
    torch.save(processed_video, os.path.join(output_tensors_dir, "all_video_tensors.pt"))
    print(f"\nSuccessfully processed and saved {video_counter} video clips.")
    return processed_video


# ==============================================================================
# 图像嵌入生成函数：使用 CLIP Vision Encoder
# ==============================================================================
def generate_video_embeddings(
        video_tensors: torch.Tensor,
        pretrained_model_path: str = "E:/store/DynaMind-main/outputs/RSM/checkpoints/clip-vit-large-patch14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[torch.Tensor]:
    """
    使用 CLIP Vision Encoder 从视频帧张量列表生成 image embeddings。

    Args:
        video_tensors (List[torch.Tensor]): 包含所有视频片段的张量列表。
        pretrained_model_path (str): 预训练模型根目录。
        device (str): 运行推理的设备（'cuda' 或 'cpu'）。

    Returns:
        Optional[torch.Tensor]: 形状为 (num_videos, 768) 的平均 image embeddings 张量。
    """
    print(f"Loading CLIP vision processor and model from {pretrained_model_path}...")
    try:
        model = CLIPModel.from_pretrained(pretrained_model_path).to(device)
        processor = CLIPImageProcessor.from_pretrained(pretrained_model_path)
        vision_encoder = model.vision_model
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check if the model path and subfolder names are correct.")
        return None

    vision_encoder.eval()

    all_video_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(video_tensors.size()[0]):
            print(f"Generating embeddings for video clip {i + 1}/{len(video_tensors)}...")
            video_clip = video_tensors[i]
            # 视频张量形状为 (F, C, H, W)，需要处理每一帧
            frame_embeddings: List[torch.Tensor] = []

            for frame_tensor in video_clip:
                # CLIP 图像处理器期望输入为 HWC 或 PIL Image
                frame_hwc = frame_tensor.permute(1, 2, 0).numpy().astype("uint8")

                processed_frame = processor(images=frame_hwc, return_tensors="pt").to(device)
                output = vision_encoder(**processed_frame)

                projected_embedding = model.visual_projection(output.pooler_output)
                frame_embeddings.append(projected_embedding)

            # 对所有帧的嵌入取平均得到视频的嵌入
            if frame_embeddings:
                video_embedding = torch.stack(frame_embeddings).mean(dim=0)
                all_video_embeddings.append(video_embedding)
            else:
                print(f"Warning: No frames found for video clip {i + 1}.")

    if all_video_embeddings:
        final_embeddings = torch.cat(all_video_embeddings, dim=0)
        return final_embeddings
    else:
        print("No embeddings were generated. Please check your input.")
        return None


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 指定包含MP4视频的目录路径
    video_data_dir = "E:/store/DynaMind-main/data/Video"

    # 第1步：处理视频并得到张量列表
    print("--- Step 1: Processing videos and saving clips ---")
    video_tensors_list = process_videos_to_tensors(video_data_dir)

    # 第2步：使用张量列表生成图像嵌入
    print("\n--- Step 2: Generating image embeddings ---")
    image_embeddings = generate_video_embeddings(video_tensors_list)

    if image_embeddings is not None:
        print("\n--- Processing Complete ---")
        print(f"Final shape of the image embeddings: {image_embeddings.shape}")

        # 第3步：保存最终的嵌入张量
        output_tensor_path = "E:/store/DynaMind-main/data/image_embs/video_image_embeddings.npy"
        os.makedirs(os.path.dirname(output_tensor_path), exist_ok=True)
        np.save(output_tensor_path, image_embeddings.cpu().numpy())
        print(f"Image embeddings saved to {output_tensor_path}")