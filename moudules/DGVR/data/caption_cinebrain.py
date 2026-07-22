import os
import cv2
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration



video_dir = "/path/to/cinebrain/dataset/clips"

output_dir = "/path/to/cinebrain/dataset/caption_order"
os.makedirs(output_dir, exist_ok=True)


local_model_path = "/path/to/DynaMind-main/moudules/RSM/checkpoints/blip"
device = "cuda"
processor = Blip2Processor.from_pretrained(local_model_path)
model = Blip2ForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
model.to(device)

def generate_caption(frames):
    """
    Generate one text description from a group of PIL image frames.
    """
    if not frames:
        return ""












    mid_frame_idx = len(frames) // 2
    image = frames[mid_frame_idx]

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)


    generated_ids = model.generate(**inputs,max_new_tokens=20, num_beams=5)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text

def process_video_and_generate_captions(video_path):
    """
    Process one video file and generate three descriptions.
    """
    print(f"Processing video: {video_path}")

    captures = cv2.VideoCapture(video_path)
    if not captures.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None, None, None


    total_frames = int(captures.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = captures.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Error: FPS is zero for video {video_path}")
        return None, None, None


    frames_per_2s = int(fps * 2)

    all_frames = []
    while True:
        ret, frame = captures.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(Image.fromarray(frame_rgb))

    captures.release()

    if len(all_frames) < total_frames:
        print(f"Warning: Extracted {len(all_frames)} frames, expected {total_frames} frames. File may be corrupted.")


    full_video_frames = all_frames
    first_2s_frames = all_frames[:frames_per_2s]
    last_2s_frames = all_frames[frames_per_2s:]


    full_caption = generate_caption(full_video_frames)
    first_2s_caption = generate_caption(first_2s_frames)
    last_2s_caption = generate_caption(last_2s_frames)

    return full_caption, first_2s_caption, last_2s_caption


if __name__ == "__main__":

    all_filenames = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    all_filenames.sort()


    full_video_captions = []
    first_2s_captions = []
    last_2s_captions = []


    for filename in all_filenames:
        print(f"Processing file: {filename}")
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, filename)


            full, first, last = process_video_and_generate_captions(video_path)

            if full and first and last:
                full_video_captions.append(full)
                first_2s_captions.append(first)
                last_2s_captions.append(last)
                print(f"Successfully generated captions for {filename}\n")


    with open(os.path.join(output_dir, "full_video_captions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(full_video_captions))

    with open(os.path.join(output_dir, "first_2s_captions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(first_2s_captions))

    with open(os.path.join(output_dir, "last_2s_captions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(last_2s_captions))

    print("\nAll captions have been saved to the output directory.")