import torch
from transformers import CLIPTokenizer, CLIPTextModel
import os
from typing import List
import numpy as np


def generate_text_embeddings_from_files(
        text_dir: str,
        output_path: str,
        pretrained_model_path: str = "E:/store/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    从指定目录下的所有txt文件中读取文本，生成 CLIP 文本嵌入，并保存。

    Args:
        text_dir (str): 包含多个txt文件的目录路径。
        output_path (str): 最终保存 embeddings 的.pt文件路径。
        pretrained_model_path (str): 预训练模型根目录。
        device (str): 运行推理的设备（'cuda' 或 'cpu'）。
    """
    # 1. 加载分词器和文本编码器
    print(f"Loading CLIP tokenizer and text encoder from {pretrained_model_path}...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check if the model path and subfolder names are correct.")
        return

    # 2. 读取所有文本文件中的内容
    all_prompts: List[str] = []
    print(f"Reading text from directory: {text_dir}...")

    # 检查路径是否存在
    if not os.path.isdir(text_dir):
        print(f"Error: Directory not found at {text_dir}")
        return

    # 遍历目录下的所有文件
    for filename in sorted(os.listdir(text_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
                all_prompts.extend(prompts)
            print(f"Read {len(prompts)} lines from {filename}. Total prompts so far: {len(all_prompts)}")

    if not all_prompts:
        print("No text lines found in the specified directory. Exiting.")
        return

    print(f"Total number of prompts to process: {len(all_prompts)}")

    # 3. 对所有文本进行分词和编码
    print("Tokenizing and encoding all prompts...")
    text_inputs = tokenizer(
        all_prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    # 4. 运行文本编码器生成 embeddings
    with torch.no_grad():
        all_embeddings = text_encoder(text_inputs)[0]

    # 5. 验证形状并保存
    print("Text embeddings generated successfully.")
    expected_shape = (len(all_prompts), 77, 768)
    if all_embeddings.shape == expected_shape:
        print(f"Final tensor shape: {all_embeddings.shape}. This matches the expected shape.")
    else:
        print(f"Warning: Final tensor shape is {all_embeddings.shape}, expected {expected_shape}.")

    # 确保保存目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存张量
    np.save(output_path, all_embeddings.cpu().numpy())
    print(f"Text embeddings saved to {output_path}")


if __name__ == "__main__":
    # 示例使用
    # 请根据你的实际路径修改这两个变量

    my_text_directory = "E:/store/DynaMind-main/data/Video/BLIP-caption"
    my_output_path = "E:/store/DynaMind-main/data/text_embs/text_embeddings.npy"

    # 调用函数
    generate_text_embeddings_from_files(my_text_directory, my_output_path)