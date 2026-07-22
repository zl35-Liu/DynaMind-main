

import os

import sys
import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from moudules.DGVR.models.unet import UNet3DConditionModel
from moudules.DGVR.data.dataset import TuneMultiVideoDataset1,TuneMultiVideoDataset2,TuneMultiVideoDataset3

from moudules.DGVR.models.pipeline_tuneavideo import TuneAVideoPipeline
from utils import *
from einops import rearrange



import torch
import torch.optim as optim
from PIL import Image
import decord
import  matplotlib.pyplot as plt
import statistics


def compute_global_mean_var(data):
    flattened_data = data.flatten()
    global_mean = np.mean(flattened_data)
    global_var = np.var(flattened_data)
    return global_mean, global_var

def save_video_frames(video_tensor, output_path):


    frames = []
    for frame in video_tensor:
        frame = frame.permute(1, 2, 0).cpu().numpy()
        frame = (((frame+1)/2) * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))


    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=125,
        loop=0
    )




os.environ["PYTORCH_CUDA_ALLOC_conf"] = "max_split_size_mb:24"


check_min_version("0.10.0.dev0")


logger = get_logger(__name__, log_level="INFO")


def main(

        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        validation_steps: int = 100,
        trainable_modules: Tuple[str] = ("attn1.to_q", "attn2.to_q", "attn_temp"),
        train_batch_size: int = 2,
        max_train_steps: int = 1200000,
        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 5.0,
        gradient_accumulation_steps: int = 2,
        gradient_checkpointing: bool = True,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        seed: Optional[int] = None,
):





    *_, config = inspect.getargvalues(inspect.currentframe())


    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if seed is not None:
        set_seed(seed)


    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))


    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", from_tf=True)
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")


    train_dataset = TuneMultiVideoDataset3(
        tokenizer=tokenizer,
        video_paths=["/path/to/DynaMind-main/data/Video/1.mp4"],
        prompt_path="/path/to/DynaMind-main/data/Video/BLIP-caption/1st_10min.txt",
        width=512,
        height=288,
        n_sample_frames=16,
        sample_frame_rate=3,
    )
    for i,data in enumerate(train_dataset.video_data):
        save_video_frames(data, f"/path/to/DynaMind-main/data/test/{i}.gif")
    print("save test data")



    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )


    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    video_list=[]
    for k in range(1,8):
        path = f'/path/to/DynaMind-main/data/Video/{k}.mp4'

        vr = decord.VideoReader(path, width=512, height=288)


        fps = 24
        total_frames = len(vr)

        waste_time = 3
        block_time = 13

        for i in range(40):

            start = int(waste_time * fps + block_time * fps * i)

            for j in range(5):

                clip_frame_length = int(2 * fps)
                start_frame = start + clip_frame_length * j + 1
                end_frame = start_frame + clip_frame_length
                if end_frame > 12480:
                    end_frame = 12480








                clip = vr.get_batch(range(start_frame, end_frame))

                clip = rearrange(clip, "f h w c -> f c h w")



                video = clip[::8]
                video_list.append(video)





    train_dataset = TuneMultiVideoDataset2(**train_data)


    video_path2 = '../SEED-DV/Video'
    video_path = ('../SEED-DV/Video/1.mp4')






    train_dataset.video=video_list

    video_text2 = '/path/to/DynaMind-main/data/Video/BLIP-caption/all.txt'


    with open(video_text2, 'r') as f:
        text_prompts2 = [line.strip() for line in f]
        text_prompts2 = text_prompts2[:400]



    train_dataset.prompt = text_prompts2
    print("data ok")

    """
    Design note for the multi-example loader: each example is a dictionary containing a two-second video and text pair, and each training iteration should pass one example to the dataset.
    """


    train_dataset.prompt_ids = tokenizer(
        list(train_dataset.prompt),
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids






























    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False
    )


    validation_pipeline = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()

    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)


    num_train_epochs = 10








    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_train_epochs, eta_min=5e-6
    )


    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)


    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")


    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")

    global_step = 0
    first_epoch = 1

    torch.cuda.empty_cache()

    latent_path="/path/to/workspace/Ljx/EEG2Video-main/EEG2Video/vae_latents"
    latents_list = []
    noise_latents_list = []
    loss_list=[]

    for epoch in tqdm(range(first_epoch, num_train_epochs + 1)):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):



            print("pixelvalues",batch['pixel_values'].shape)

            print("promptids",batch['prompt_ids'].shape)

            with accelerator.accumulate(unet):

                pixel_values = batch["pixel_values"].to(weight_dtype)
                print(pixel_values.shape)
                video = pixel_values/2 + 0.5
                video = video.squeeze(0)
                print("video sahpe ",video.shape)
                prompts = batch["prompt_ids"]

                input_path = f"{output_dir}/input_samples"
                if not os.path.exists(input_path):
                    os.makedirs(input_path, exist_ok=True)
                if epoch==1 and step>200:
                    save_video_frames(video, f"{input_path}/{text_prompts2[step]}.gif")

                video_length = pixel_values.shape[1]

                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                print("Whether pixels are normalized: ",pixel_values.min(), pixel_values.max())


                latents = vae.encode(pixel_values).latent_dist.sample()

                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                if epoch == num_train_epochs:
                    latents_list.append(latents)







                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()







                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


                if epoch == num_train_epochs:
                    noise_latents_list.append(noisy_latents)


                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
                print(encoder_hidden_states.shape)
                out = encoder_hidden_states
                print(f"Text input range: [{out.min():.2f}, {out.max():.2f}]")
                print(f"Text embedding norm: {out.norm(dim=-1).mean():.2f}")
                out = out.cpu().detach().numpy()
                text_mean, text_var = compute_global_mean_var(out)
                print("Overall text embedding mean:", text_mean)
                print("Overall text embedding variance:", text_var)



                if noise_scheduler.prediction_type == "epsilon":
                    target = noise

                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")


                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                print(f"epoch {epoch} Processing step {step} loss {loss.item()} lr {lr_scheduler.get_last_lr()[0]}")



                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                loss_list.append(train_loss)


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()

                optimizer.zero_grad()



            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0



            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

        lr_scheduler.step()


        if epoch % 1 == 0:
            if accelerator.is_main_process:
                samples = []
                generator = torch.Generator(device=latents.device).manual_seed(seed)


                ddim_inv_latent = None
                if validation_data.use_inv_latent:
                    inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{epoch}.pt")
                    ddim_inv_latent = ddim_inversion(
                        validation_pipeline, ddim_inv_scheduler,
                        video_latent=latents,
                        num_inv_steps=validation_data.num_inv_steps,
                        prompt=""
                    )[-1].to(weight_dtype)



                for idx, prompt in enumerate(validation_data.prompts):
                    sample = validation_pipeline(
                        prompt,
                        generator=generator,
                        latents=None,
                        **validation_data
                    ).videos
                    save_videos_grid(sample, f"{output_dir}/samples/sample-{epoch}/{prompt}.gif")
                    samples.append(sample)


                samples = torch.concat(samples)
                save_path = f"{output_dir}/samples/sample-{epoch}.gif"
                save_videos_grid(samples, save_path)
                logger.info(f"Saved samples to {save_path}")


            accelerator.wait_for_everyone()


            if accelerator.is_main_process:
                unet = accelerator.unwrap_model(unet)
                pipeline = TuneAVideoPipeline.from_pretrained(
                    pretrained_model_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                )
                pipeline.save_pretrained(output_dir)


















    """
    Experiment log for saved tensor and noise variants across multiple training runs, including frame alignment, noise settings, restricted diffusion timesteps, inference alignment, and checkpoint epochs.
    """
    print("torch_save_ok")



    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)


    meandata = []

    for i in range(num_train_epochs):
        avg = statistics.mean(loss_list[i * (len(train_dataloader) / train_batch_size):(i + 1) * (
                    len(train_dataloader) / train_batch_size)])
        meandata.append(avg)
    plt.plot(range(meandata), meandata, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(f"{output_dir}/train_loss.gif")
    plt.show()

    accelerator.end_training()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/path/to/DynaMind-main/configs/dgvr.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))


"""

"""