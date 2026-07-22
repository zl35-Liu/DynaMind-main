import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from PIL import Image




video_dir = "/path/to/cinebrain/dataset/clips"
output_file = "/path/to/cinebrain/dataset/captions_qwen.txt"
model_path = "/path/to/DynaMind-main/moudules/RSM/checkpoints/qwen-vl"




device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()




import cv2

def sample_frame(video_path, frame_id=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    else:
        return None

































def video_to_text(video_path):
    frame = sample_frame(video_path, frame_id=0)
    if frame is None:
        return "Failed to extract frame."


    prompt = "Describe the scene in detail."

    inputs = processor(
        text=[prompt],
        images=[frame],
        return_tensors="pt",
    ).to(device)
    images = processor(

    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=36,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if prompt in caption:
        caption = caption.split(prompt, 1)[-1].strip()

    return caption





with open(output_file, "w", encoding="utf-8") as f:
    for i in range(8100):
        video_name = f"{i:06d}.mp4"
        video_path = os.path.join(video_dir, video_name)

        if not os.path.exists(video_path):
            print(f"File does not exist: {video_path}")
            f.write("File missing\n")
            continue

        try:
            caption = video_to_text(video_path)
        except Exception as e:
            caption = f"Error: {e}"

        f.write(caption.strip() + "\n")
        print(f"[{i}] {video_name}: {caption}")
