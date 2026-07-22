import os
import decord
import torch
from transformers import CLIPModel, CLIPImageProcessor
from einops import rearrange
from typing import List, Optional
import cv2
import numpy as np





def process_videos_to_tensors(
        video_dir: str,
        sub_dir: str = "video_tensor",
        output_height: int = 288,
        output_width: int = 512
) -> torch.Tensor:
    """
    Split and sample the seven MP4 videos in a directory, save short clips, and return the processed frame tensors.
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





def generate_video_embeddings(
        video_tensors: torch.Tensor,
        pretrained_model_path: str = "/path/to/DynaMind-main/outputs/RSM/checkpoints/clip-vit-large-patch14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[torch.Tensor]:
    """
    Generate image embeddings from video-frame tensors with a CLIP vision encoder and save the resulting features.
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

            frame_embeddings: List[torch.Tensor] = []

            for frame_tensor in video_clip:

                frame_hwc = frame_tensor.permute(1, 2, 0).numpy().astype("uint8")

                processed_frame = processor(images=frame_hwc, return_tensors="pt").to(device)
                output = vision_encoder(**processed_frame)

                projected_embedding = model.visual_projection(output.pooler_output)
                frame_embeddings.append(projected_embedding)


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





if __name__ == "__main__":

    video_data_dir = "/path/to/DynaMind-main/data/Video"


    print("--- Step 1: Processing videos and saving clips ---")
    video_tensors_list = process_videos_to_tensors(video_data_dir)


    print("\n--- Step 2: Generating image embeddings ---")
    image_embeddings = generate_video_embeddings(video_tensors_list)

    if image_embeddings is not None:
        print("\n--- Processing Complete ---")
        print(f"Final shape of the image embeddings: {image_embeddings.shape}")


        output_tensor_path = "/path/to/DynaMind-main/data/image_embs/video_image_embeddings.npy"
        os.makedirs(os.path.dirname(output_tensor_path), exist_ok=True)
        np.save(output_tensor_path, image_embeddings.cpu().numpy())
        print(f"Image embeddings saved to {output_tensor_path}")