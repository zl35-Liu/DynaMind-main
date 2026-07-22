












































































































































import os
import sys
import torch
from einops import rearrange
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from diffusers import AutoencoderKL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils.data import split_cine_into_clips


def main(
    pretrained_model_path: str,
    video_folder: str,
    output_dir: str,
    device: str = "cuda",
    vae_subfolder: str = "vae",
    batch_size: int = 8,
    sample_rate: int = 3,
    total_frames_per_clip: int = 16,
    resize_h: int = 288,
    resize_w: int = 512,
    save_per_batch: bool = True,
):
    """
    Load all videos from a directory, sample and resize their frames, and encode the resulting clips with the VAE.
    """


    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    video_files.sort()
    print(f"Detected  {len(video_files)}。")

    if len(video_files) == 0:
        raise ValueError(f"No video files were found in {video_folder} .")

    os.makedirs(output_dir, exist_ok=True)


    print("\n--- Loading VAE model ---")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder=vae_subfolder)
    vae.to(device).eval()
    vae.requires_grad_(False)


    latent_idx = 0
    all_latents = []

    for idx, video_path in enumerate(tqdm(video_files, desc="Processing video clips")):
        try:
            clips = split_cine_into_clips(
                video_path=video_path,
                sample_rate=sample_rate,
                final_frames_per_clip=total_frames_per_clip,
                resize_h=resize_h,
                resize_w=resize_w,
            )
            if len(clips) == 0:
                continue

            all_video_tensor = torch.stack(clips)
            dataset = TensorDataset(all_video_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for batch_idx, batch in enumerate(dataloader):
                clips_batch = batch[0]

                pixel_values = (clips_batch / 127.5 - 1.0).to(device).float()
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()

                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                if save_per_batch:
                    save_path = os.path.join(output_dir, f"latent_{latent_idx:05d}.pt")
                    torch.save(latents.cpu(), save_path)
                    latent_idx += 1
                else:
                    all_latents.append(latents.cpu())

        except Exception as e:
            print(f"❌ Failed to process  {video_path}  failed: {e}")
            continue


    if not save_per_batch:
        all_latents = torch.cat(all_latents, dim=0)
        final_path = os.path.join(output_dir, "all_latents.pt")
        torch.save(all_latents, final_path)
        print(f"\n✅ All latent representations were saved to: {final_path}")
    else:
        print(f"\n✅ All latent representations were saved in batches to  {output_dir}")


if __name__ == "__main__":
    PRETRAINED_MODEL_PATH = "/path/to/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4"
    VIDEO_FOLDER = "/path/to/cinebrain/dataset/clips"
    OUTPUT_DIR = "/path/to/DynaMind-main/data/vae_latents/cine6"

    main(
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        video_folder=VIDEO_FOLDER,
        output_dir=OUTPUT_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        sample_rate=5,
        total_frames_per_clip=6,
        resize_h=480,
        resize_w=720,
        save_per_batch=True,
    )
