import torch
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
import os
from moudules.DGVR.models.unet import UNet3DConditionModel
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import logging
import decord


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
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
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
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
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

    parser = argparse.ArgumentParser(description="DDIM Inversion for Video Latents")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--vae_latents_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="ddim_inverted_latents.pt")
    parser.add_argument("--num_inv_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])


    args = argparse.Namespace()
    args.pretrained_model_path = "/path/to/stable-diffusion-v1-4"
    args.vae_latents_path = ""
    args.output_path = "/path/to/DynaMind-main/outputs/TDA/video_noise/dv16/ddim_inv_latents_16_first200.pt"
    args.num_inv_steps = 50
    args.batch_size = 2
    args.dtype = "float32"

    dtype = torch.float32 if args.dtype == "float32" else torch.float16


    logger.info("Loading pretrained model...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", from_tf=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder = text_encoder.float().to(dtype)
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet").to(device)
    unet = unet.float().to(dtype)
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")


    scheduler.set_timesteps(args.num_inv_steps)


    video_list=[]
    for k in range(1,2):
        path = f"/path/to/eegvideo/data/Video/v/{k}.mp4"

        vr = decord.VideoReader(path, width=512, height=288)


        fps = 24
        total_frames = len(vr)

        waste_time=3
        block_time=13

        for i in range(40):

            start = int(waste_time * fps+block_time*fps*i)

            for j in range(5):


                clip_frame_length = int(2 * fps )
                start_frame = start + clip_frame_length*j+1
                end_frame = start_frame + clip_frame_length
                if end_frame >12480:
                    end_frame = 12480








                clip = vr.get_batch(range(start_frame, end_frame))

                clip = torch.from_numpy(clip.asnumpy())
                clip = rearrange(clip, "f h w c -> f c h w")



                video = clip[::3]
                video = video/127.5 - 1
                video_list.append(video)
    video_list=torch.stack(video_list).to(device).to(dtype)

















































    inv_latents = []

    logger.info(f"Starting DDIM inversion ({args.num_inv_steps} steps)...")


    batch_size = args.batch_size
    num_videos = video_list.shape[0]

    for start_idx in tqdm(range(0, num_videos, batch_size)):
        end_idx = min(start_idx + batch_size, num_videos)
        batch_videos = video_list[start_idx:end_idx]


        with torch.no_grad():
            b, f, c, h, w = batch_videos.shape
            batch_videos = rearrange(batch_videos, "b f c h w -> (b f) c h w")
            latents = vae.encode(batch_videos).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b, f=f)
            latents = latents * 0.18215


        for j in range(latents.shape[0]):
            video_latent = latents[j:j+1].to(device).to(dtype)
            ddim_latents = ddim_inversion(
                ddim_scheduler=scheduler,
                video_latent=video_latent,
                num_inv_steps=args.num_inv_steps,
                prompt="",
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                device=device
            )
            inv_latent = ddim_latents[-1].cpu()
            inv_latents.append(inv_latent)


    inv_latents = torch.cat(inv_latents, dim=0)
    logger.info(f"Inverted latent shape: {inv_latents.shape}")

    torch.save(inv_latents, args.output_path)
    logger.info(f"Saving inverted latents to: {args.output_path}")
