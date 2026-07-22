import torch
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from moudules.DGVR.models.unet import UNet3DConditionModel

from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer




from diffusers import CogVideoXPipeline

import argparse
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def init_prompt(prompt, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel, device, max_seq_len=226):
    '''
    for CogVideoX, we use T5 encoder for text embeddings.
    '''

    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]


    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]




    context = torch.cat([uncond_embeddings, text_embeddings])
    return context




















def next_step(model_output, timestep, sample, ddim_scheduler):
    timestep, next_timestep = min(







        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999
    ), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]


    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample



@torch.no_grad()
def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred



@torch.no_grad()
def ddim_loop(ddim_scheduler, latent, num_inv_steps, prompt, tokenizer, text_encoder, unet, device):
    context = init_prompt(prompt, tokenizer, text_encoder, device)
    _, cond_embeddings = context.chunk(2)




    latent = latent.clone().detach()


    for i in range(num_inv_steps):

        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]


        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, unet)




        latent = next_step(noise_pred, t, latent, ddim_scheduler)



    return latent


@torch.no_grad()
def ddim_inversion(ddim_scheduler, video_latent, num_inv_steps, prompt, tokenizer, text_encoder, unet, device):

    final_latent_xt = ddim_loop(ddim_scheduler, video_latent, num_inv_steps, prompt, tokenizer, text_encoder, unet, device)
    return final_latent_xt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDIM Inversion for VAE Latents")
    parser.add_argument("--pretrained_model_path", type=str, default='/path/to/cogvideo/checkpoints/CogVideoX-5b')
    parser.add_argument("--vae_latents_dir", type=str, default='/path/to/DynaMind-main/data/vae_latents/cine_cog/200per/')
    parser.add_argument("--output_dir", type=str, default="/path/to/DynaMind-main/data/ddim_invs/cine_cog/")
    parser.add_argument("--caption_file", type=str, default="/path/to/cinebrain/dataset/captions_simplified.txt", help="Text file containing all 8,100 English sentences")
    parser.add_argument("--num_inv_steps", type=int, default=50)

    parser.add_argument("--save_chunk_size", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16"])
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    os.makedirs(args.output_dir, exist_ok=True)


    logger.info("Loading pretrained model components...")







    pipe = CogvideoXPipeline.from_pretrained(args.pretrained_model_path, torch_dtype=dtype)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device, dtype)

    unet = CogVideoXTransformer3DModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet").to(device, dtype)

    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inv_steps)

    text_encoder.eval()
    unet.eval()


    logger.info(f"Loading VAE latent file path: {args.vae_latents_dir}")

    vae_names = sorted(os.listdir(args.vae_latents_dir), key = lambda x: int(x.split('_')[-1].split('.')[0]))
    latent_file_paths = [os.path.join(args.vae_latents_dir, f) for f in vae_names if f.endswith(".pt") or f.endswith(".pth")]
    print(latent_file_paths[:20])
    if len(latent_file_paths) == 0:
        raise FileNotFoundError("No PT or PTH files were found at the specified path")

    logger.info(f"Detected  {len(latent_file_paths)}  merged latent files")


    with open(args.caption_file, "r", encoding="utf-8") as f:
        all_captions = [line.strip() for line in f.readlines()]
    logger.info(f"Loaded caption file with  {len(all_captions)}  descriptions")


    if len(all_captions) % args.save_chunk_size != 0:
         logger.warning("The caption count is not an integer multiple of the latent file size; verify data alignment.")


    all_inv_latents_chunk = []
    global_latent_index = 0


    for file_index, latent_path in enumerate(tqdm(latent_file_paths, desc="Processing Latent Files")):

        latents_chunk = torch.load(latent_path, map_location=device).to(dtype)
        print(f"Loaded latent chunk shape: {latents_chunk.shape} from {latent_path}")


        if latents_chunk.ndim != 5:

            if latents_chunk.ndim == 4:
                latents_chunk = latents_chunk.unsqueeze(0)
            else:
                 raise ValueError(f"Incorrect latent dimensions: {latents_chunk.ndim}. Expected five dimensions (N, C, T, H, W).")

        num_latents_in_file = latents_chunk.shape[0]
        logger.info(f"Processing file  {latent_path}, containing  {num_latents_in_file}  latents")


        for i in range(num_latents_in_file):
            current_latent = latents_chunk[i].unsqueeze(0)



            if global_latent_index >= len(all_captions):
                logger.error(f"Insufficient captions. Latent index: {global_latent_index}")
                break

            current_prompt = all_captions[global_latent_index]


            final_inv_latent = ddim_inversion(
                ddim_scheduler=scheduler,
                video_latent=current_latent,
                num_inv_steps=args.num_inv_steps,
                prompt="",
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                device=device
            )


            all_inv_latents_chunk.append(final_inv_latent.cpu())

            global_latent_index += 1



        if all_inv_latents_chunk:

            chunk_path = os.path.join(args.output_dir, f"{file_index:03d}.pt")
            torch.save(torch.cat(all_inv_latents_chunk, dim=0), chunk_path)
            logger.info(f"✅ Saved DDIM inversion result file  {file_index:03d}.pt, containing  {len(all_inv_latents_chunk)}  segments")
            all_inv_latents_chunk.clear()

    logger.info("🎉 All DDIM inversions are complete.")