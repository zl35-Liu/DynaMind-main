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
import argparse
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@torch.no_grad()
def init_prompt(prompt, tokenizer, text_encoder, device):
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
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
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in range(num_inv_steps):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(ddim_scheduler, video_latent, num_inv_steps, prompt, tokenizer, text_encoder, unet, device):
    ddim_latents = ddim_loop(ddim_scheduler, video_latent, num_inv_steps, prompt, tokenizer, text_encoder, unet, device)
    return ddim_latents



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDIM Inversion for VAE Latents")
    parser.add_argument("--pretrained_model_path", type=str, default= '/path/to/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4')
    parser.add_argument("--vae_latents_dir", type=str, default='/path/to/DynaMind-main/data/vae_latents/cine6')
    parser.add_argument("--output_dir", type=str, default="/path/to/DynaMind-main/data/ddim_invs/cine6")
    parser.add_argument("--caption_file", type=str, default="/path/to/cinebrain/dataset/captions_simplified.txt",
                        help="Text file containing 8,100 English sentences")
    parser.add_argument("--num_inv_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_chunk_size", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16"])
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    os.makedirs(args.output_dir, exist_ok=True)


    logger.info("Loading pretrained model components...")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder = text_encoder.to(dtype)
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet").to(device)
    unet = unet.to(dtype)
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inv_steps)

    text_encoder.eval()
    unet.eval()


    logger.info(f"Loading VAE latent file path: {args.vae_latents_dir}")
    latent_files = sorted([os.path.join(args.vae_latents_dir, f)
                           for f in os.listdir(args.vae_latents_dir) if f.endswith(".pt") or f.endswith(".pth")])
    print(latent_files[:20])

    if len(latent_files) == 0:
        raise FileNotFoundError("No PT or PTH files were found at the specified path")

    logger.info(f"Detected  {len(latent_files)}  latent files")


    with open(args.caption_file, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f.readlines()]
    logger.info(f"Loaded caption file with  {len(captions)}  descriptions")

    if len(captions) != len(latent_files):
        raise ValueError(f"Latent count ({len(latent_files)}) and caption count ({len(captions)}) do not match.")



    all_inv_latents = []
    global_index = 0
    chunk_index = 0

    for latent_path, prompt in tqdm(zip(latent_files, captions), total=len(latent_files)):
        latents = torch.load(latent_path, map_location=device).to(dtype)
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        logger.info(f"Processing  {latent_path}, shape={tuple(latents.shape)}")



        ddim_latents = ddim_inversion(
            ddim_scheduler=scheduler,
            video_latent=latents,
            num_inv_steps=args.num_inv_steps,
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            device=device
        )
        print(f"✅ Completed DDIM inversion for video  {global_index} ")
        inv_latent = ddim_latents[-1].cpu()
        all_inv_latents.append(inv_latent)
        global_index += 1


        if global_index % args.save_chunk_size == 0 or global_index == len(latent_files):
            chunk_path = os.path.join(args.output_dir, f"{chunk_index:03d}.pt")
            torch.save(torch.cat(all_inv_latents, dim=0), chunk_path)
            logger.info(f"✅ Saved file  {chunk_index} , containing  {len(all_inv_latents)}  segments ->  {chunk_path}")
            all_inv_latents.clear()
            chunk_index += 1

    logger.info("🎉 All DDIM inversions are complete.")
