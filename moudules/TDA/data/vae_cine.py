import os
import glob
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Union


from diffusers import DiffusionPipeline, CogVideoXPipeline





COGVIDEOX_MODEL_ID = "/path/to/cogvideo/checkpoints/CogVideoX-5b"


try:
    from decord import VideoReader
    from decord import cpu, gpu

    def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:

        frames_tensor = torch.from_numpy(frames).float()


        if frames_tensor.max() > 1.0:
            frames_tensor /= 255.0


        frames_tensor = 2 * frames_tensor - 1.0


        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)
        return frames_tensor

    def load_and_preprocess_video(
        video_path: str,
        target_frames: int,
        target_size: tuple
    ) -> Union[torch.Tensor, None]:
        """
        Load and sample a video, then convert its frames to the range from negative one to one expected by the VAE.
        """
        try:

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)


            if total_frames < target_frames:
                print(f"Skipping {video_path}: Only {total_frames} frames found (Need {target_frames}).")
                return None


            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()


            video_tensor = frames_to_tensor(frames)




            return video_tensor

        except Exception as e:
            print(f"Error loading video {video_path} with decord: {e}")
            return None

except ImportError:
    print("Error: 'decord' not installed. Please install it (`pip install decord`).")

    def load_and_preprocess_video(*args, **kwargs):
        raise ImportError("Video loading function requires 'decord'.")



def batch_encode_videos_to_latent(
    vae_model,
    video_dir: str,
    output_dir: str,
    videos_per_file: int = 200,
    target_frames: int = 16,
    target_size: tuple = (720, 480),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load videos in batches, encode them with the VAE, and save each batch.
    """

    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi', '*.webm']:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))

    if not video_files:
        print(f"No video files found in {video_dir}. Exiting.")
        return

    total_videos = len(video_files)
    print(f"Found {total_videos} video files.")


    os.makedirs(output_dir, exist_ok=True)

    vae_model.to(device).eval()

    all_latents_batch: List[torch.Tensor] = []
    file_counter = 0
    encoded_count = 0


    for i, video_path in enumerate(tqdm(video_files, desc="Encoding Videos")):

        video_tensor = load_and_preprocess_video(video_path, target_frames, target_size)

        if video_tensor is None:
            continue

        try:

            video_tensor = video_tensor.to(device)


            with torch.no_grad():

                latent_dist = vae_model.encode(video_tensor).latent_dist



                vae_latent = latent_dist.sample().cpu()

            all_latents_batch.append(vae_latent)
            encoded_count += 1


            if len(all_latents_batch) >= videos_per_file:

                merged_latents = torch.cat(all_latents_batch, dim=0)


                file_counter += 1
                save_path = os.path.join(output_dir, f"latents_batch_{file_counter:04d}.pt")
                torch.save(merged_latents, save_path)


                all_latents_batch = []
                tqdm.write(f"Saved batch {file_counter} ({videos_per_file} latents) to {save_path}")

        except Exception as e:
            tqdm.write(f"\nFailed to encode {video_path}. Error: {e}")


    if all_latents_batch:
        file_counter += 1
        merged_latents = torch.cat(all_latents_batch, dim=0)
        save_path = os.path.join(output_dir, f"latents_batch_{file_counter:04d}.pt")
        torch.save(merged_latents, save_path)
        tqdm.write(f"Saved final batch {file_counter} ({len(all_latents_batch)} latents) to {save_path}")

    print(f"\n--- Batch Encoding Complete ---")
    print(f"Total encoded videos: {encoded_count}")
    print(f"Total files saved: {file_counter}")
    print(f"Latents saved to: {output_dir}")


if __name__ == "__main__":

    VIDEO_DIRECTORY = "/path/to/cinebrain/dataset/clips"
    OUTPUT_DIRECTORY = "/path/to/DynaMind-main/data/vae_latents/cine_cog"


    try:






        print(f"Loading VAE from model: {COGVIDEOX_MODEL_ID}...")







        pipe = CogVideoXPipeline.from_pretrained(
            COGVIDEOX_MODEL_ID,
            torch_dtype=torch.float16
        )
        vae = pipe.vae





        print("Using SVD's VAE for demonstration. Please replace with actual CogVideoX VAE.")

    except Exception as e:
        print(f"Failed to load VAE: {e}")
        print("Exiting script.")
        exit()


    batch_encode_videos_to_latent(
        vae_model=vae,
        video_dir=VIDEO_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        videos_per_file=200,
        target_frames=32,
        target_size=(720, 480)
    )