










































































































import torch
from transformers import CLIPTokenizer, CLIPTextModel
import os
from typing import List
import numpy as np
from tqdm.auto import tqdm


def generate_text_embeddings_from_files(
        text_dir: str,
        output_path: str,
        pretrained_model_path: str = "/path/to/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4",
        batch_size: int = 64,
        dtype: str = "float16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Read text from one TXT file or a directory of TXT files, generate CLIP text embeddings in batches, and save them as an NPY or PT file.
    """

    print(f"Loading CLIP tokenizer and text encoder from {pretrained_model_path}...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)

    if dtype == "float16":
        text_encoder = text_encoder.half()


    all_prompts: List[str] = []
    if os.path.isdir(text_dir):
        print(f"Reading all .txt files under directory: {text_dir}")
        for filename in sorted(os.listdir(text_dir)):
            if filename.endswith(".txt"):
                with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    all_prompts.extend(lines)
    else:
        with open(text_dir, "r", encoding="utf-8") as f:
            all_prompts = [line.strip() for line in f if line.strip()]

    print(f"✅ Total number of prompts: {len(all_prompts)}")


    all_embeddings = []
    print(f"Generating embeddings in batches of {batch_size}...")

    for i in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i: i + batch_size]


        inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)


        with torch.no_grad():
            batch_emb = text_encoder(inputs.input_ids)[0]
        all_embeddings.append(batch_emb.cpu())


        del batch_emb, inputs
        torch.cuda.empty_cache()


    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"✅ Final embeddings shape: {all_embeddings.shape}")


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".pt"):
        torch.save(all_embeddings, output_path)
    else:
        np.save(output_path, all_embeddings.numpy())

    print(f"💾 Text embeddings saved to: {output_path}")


if __name__ == "__main__":
    my_text_directory = "/path/to/cinebrain/dataset/captions_simplified.txt"
    my_output_path = "/path/to/DynaMind-main/data/text_embs/cine/text_embeddings.pt"

    generate_text_embeddings_from_files(
        text_dir=my_text_directory,
        output_path=my_output_path,
        pretrained_model_path="/path/to/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4",
        batch_size=64,
        dtype="float16",
        device="cuda"
    )