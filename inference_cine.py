



import sys
import os
from random import random
import random
import torch



torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from triton.language import dtype

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from moudules.DGVR.models.pipeline import TuneAVideoPipeline
from moudules.DGVR.models.unet import UNet3DConditionModel
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from utils.visualize import save_videos_grid
import torch
from torch import nn


import numpy as np
from einops import rearrange
from sklearn import preprocessing
from transformers import CLIPTextModel, CLIPTokenizer

class CLIP1(nn.Module):
    def __init__(self,input_dim):
        super(CLIP1, self).__init__()
        self.input_dim = input_dim
    def forward(self, x):
        return x

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:

        print('Note: not using cudnn.deterministic')
seed_everything(114514)




pretrained_eeg_encoder_path = '/path/to/workspace/Ljx/EEG2Video-main/EEG2Video/models/semantic_predictor_f3.pt'



model = CLIP1(77*768)


model.to(torch.device('cuda'))
model.eval()


eeg_data_path = "/path/to/DynaMind-main/outputs/RSM/contra_embs1.npy"

text_path = '/path/to/DynaMind-main/data/text_embs/cine/text_embeddings.npy'
block_num = 1
sub = 1



EEG_dim = 62*5
eegdata = np.load(eeg_data_path)

textdata = np.load(text_path)



print("eeg and text shape",textdata.shape)


















































textdata = torch.from_numpy(textdata)
textdata = textdata.half()
textdata = textdata.to(torch.device('cuda'))



























pretrained_model_path = "/path/to/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4"
my_model_path = "/path/to/DynaMind-main/outputs/DGVR/checkpoints/tuned/cine16/4/1"


unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet',use_memory_efficient_attention=True,local_files_only=True,torch_dtype=torch.float16).to('cuda')




generator = torch.Generator(device=torch.device('cuda')).manual_seed(33)
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path,generator=generator, unet=unet, torch_dtype=torch.float16).to("cuda")

print("Pipeline components keys:", pipe.components.keys())
print(hasattr(pipe, 'text_encoder'))

pipe.enable_vae_slicing()

















"""
Experimental notes for loading precomputed latent variants without DANA. The listed alternatives compare prepared Seq2Seq latents, random latents, and several dataset-specific latent files.
"""
















inv_latent = torch.load('/path/to/DynaMind-main/data/ddim_invs/cine6/000.pt',map_location='cpu')



print("inv shape",inv_latent.shape)


woSeq2Seq = False
woDANA = False

negative = textdata.mean(dim=0)

print("negative shape",negative.shape)

for i in range(200):
    if i % 5 != 4 or i < 10:
        continue
    print("Video ",i,"")
    if woSeq2Seq:
        video = pipe(model, eeg_test[i:i+1,...], latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = '40_Classes_woSeq2Seq'
    elif woDANA:

        video = pipe(model,textdata[i:i+1,...], latents=inv_latent[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=50, guidance_scale=12.5).videos
        savename = '40_Classes_woDANA37'
    else:
        video = pipe(model, textdata[i:i+1,...].to(dtype=pipe.unet.dtype, device=pipe.device),negative_eeg=negative, latents=inv_latent[i:i+1,...].to(dtype=pipe.unet.dtype, device=pipe.device), video_length=6, height=480, width=720, num_inference_steps=50, guidance_scale=12.5).videos
        savename = ('0')
    save_videos_grid(video, f"./outputs/reconstruction/cine6/{savename}/{i}.gif")



"""
Experiment log for Cine inference variants, covering checkpoint epochs, DDIM step counts, text embeddings, random initial latents, continuity, visual artifacts, and scene transitions.
"""
