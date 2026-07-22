import os
import cv2
import torch
import shutil
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info




quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)



video_dir = "/path/to/cinebrain/dataset/clips_split/first/"

output_dir = "/path/to/cinebrain/dataset/caption_qwen25/"
os.makedirs(output_dir, exist_ok=True)



local_model_path = "/path/to/DynaMind-main/moudules/RSM/checkpoints/qwen2.5-vl/"
device = "cuda"


print(f"Loading Qwen2.5-VL model from: {local_model_path}")
try:

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,

        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    ).eval()


    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and Processor loaded successfully.")

except Exception as e:
    print(f"Error loading Qwen2.5-VL model: {e}")
    print("FATAL: ensure qwen_vl_utils is installed and all Qwen2.5-VL files are complete.")
    exit()

def generate_video_caption(video_path, prompt_text, video_fps=1):
    """
    Generate a video description with the official Qwen2.5-VL workflow.
    """


    video = cv2.VideoCapture(video_path)
    messages = [
        {
            "role": "user",
            "content": [

                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]





    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )



    image_inputs, video_inputs = process_vision_info(messages)


    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,

        padding=False,
        video_fps=video_fps,
        return_tensors="pt",
    )


    inputs = {k: v.to(device) for k, v in inputs.items()}


    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_new_tokens=48,


    )


    input_ids_length = inputs["input_ids"].shape[1]
    generated_ids_trimmed = [
        out_ids[input_ids_length:] for out_ids in generated_ids
    ]


    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0].strip()

    return output_text


if __name__ == "__main__":


    all_filenames = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    all_filenames.sort()
    print(f"Found {len(all_filenames)} videos in {video_dir}")
    print(all_filenames[:20])

    full_video_captions = []


    video_prompt = "Generate a detailed and chronological description of the main events and actions in the video using up to 30 words."

    for filename in all_filenames:
        video_path = os.path.join(video_dir, filename)

        print(f"\nProcessing video: {video_path}")

        try:

            full_caption = generate_video_caption(video_path, video_prompt, video_fps=1)

            if full_caption:
                full_video_captions.append(full_caption)
                print(f"SUCCESS for {filename}")
                print(f"Caption: {full_caption}...")
            else:
                full_video_captions.append("")
                print(f"Warning: Failed to generate caption for {filename}")

        except Exception as e:


            import traceback

            traceback.print_exc()

            error_message = f"ERROR: {type(e).__name__}: {str(e)}"
            full_video_captions.append(error_message)
            print(f"An unexpected error occurred while processing {filename}:")
            print(error_message)
            print("--- Full Stack Trace Above ---")


    with open(os.path.join(output_dir, "first_captions_qwen2_5.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(full_video_captions))

    print("\nAll captions have been saved to the output directory.")
