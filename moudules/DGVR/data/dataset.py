import os
import decord


decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):

        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)

        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]

        print(len(sample_index))

        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")


        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example



class TuneMultiVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: list,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 8,

            block: int = 40,
            clips: int = 5,
            waste_time: float = 3.0,
            clip_duration: float = 10.0,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate=sample_frame_rate

        self.block= block
        self.clips = clips
        self.waste_time = waste_time
        self.clip_duration = clip_duration

    def __len__(self):

        return self.block * self.clips

    def __getitem__(self, index):

        try:
            vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        except Exception as e:
            print(f"Error loading video {self.video_path}: {e}")

        fps = 24
        total_frames = len(vr)


        block_time=self.waste_time+self.clip_duration

        total=[]
        for i in range(self.block):

            start = int(self.waste_time * fps+block_time*fps*i)

            for j in range(5):


                clip_frame_length = int(2 * fps )
                start_frame = start + clip_frame_length*j+1
                end_frame = start_frame + clip_frame_length
                if end_frame >12480:
                    end_frame = 12480




                if end_frame > total_frames+1:
                    raise ValueError(f"Clip {index} exceeds video length.")


                clip = vr.get_batch(range(start_frame, end_frame))

                clip = rearrange(clip, "f h w c -> f c h w")



                video = clip[::self.sample_frame_rate]



                example = {
                    "pixel_values": (video / 127.5 - 1.0),
                    "prompt_ids": self.prompt_ids[i*5+j]
                }

                total.append(example)

        return total




class TuneMultiVideoDataset1(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: list,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 8,

            block: int = 40,
            clips: int = 5,
            waste_time: float = 3.0,
            clip_duration: float = 10.0,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate=sample_frame_rate

        self.block= block
        self.clips = clips
        self.waste_time = waste_time
        self.clip_duration = clip_duration

    def __len__(self):

        return self.block * self.clips

    def __getitem__(self, index):
        total = []
        subnames = os.listdir(self.video_path)
        for subname in subnames:
            subpath = os.path.join(self.video_path, subname)
            print(subpath)

            try:
                vr = decord.VideoReader(subpath, width=self.width, height=self.height)
            except Exception as e:
                print(f"Error loading video {self.video_path}: {e}")

            fps = 24
            total_frames = len(vr)


            block_time=self.waste_time+self.clip_duration


            for i in range(self.block):

                start = int(self.waste_time * fps+block_time*fps*i)

                for j in range(5):


                    clip_frame_length = int(2 * fps )
                    start_frame = start + clip_frame_length*j+1
                    end_frame = start_frame + clip_frame_length
                    if end_frame >12480:
                        end_frame = 12480




                    if end_frame > total_frames+1:
                        raise ValueError(f"Clip {index} exceeds video length.")


                    clip = vr.get_batch(range(start_frame, end_frame))

                    clip = rearrange(clip, "f h w c -> f c h w")



                    video = clip[::self.sample_frame_rate]



                    example = {
                        "pixel_values": (video / 127.5 - 1.0),
                        "prompt_ids": self.prompt_ids[i*5+j]
                    }

                    total.append(example)

        return total



class TuneMultiVideoDataset2(Dataset):
    def __init__(
            self,
            video: list,
            prompt: list,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 8,

            block: int = 40,
            clips: int = 5,
            waste_time: float = 3.0,
            clip_duration: float = 10.0,
    ):
        self.video = video
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate=sample_frame_rate

        self.block= block
        self.clips = clips
        self.waste_time = waste_time
        self.clip_duration = clip_duration

    def __len__(self):

        return len(self.video)

    def __getitem__(self, index):


        example = {
            "pixel_values": (self.video[index] / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[index]
        }
        return example


import os
import decord
import torch
from einops import rearrange
from torch.utils.data import Dataset

class TuneMultiVideoDataset3(Dataset):
    def __init__(
            self,
            video_paths: list,
            prompt_path: str,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_frame_rate: int = 8,
            block: int = 40,
            clips: int = 5,
            waste_time: float = 3.0,
            clip_duration: float = 10.0,
            tokenizer=None,
            model_max_length: int = 77
    ):
        self.video_paths = video_paths
        self.prompt_path = prompt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.fps = 24

        self.block = block
        self.clips = clips
        self.waste_time = waste_time
        self.clip_duration = clip_duration


        self.tokenizer = tokenizer
        self.model_max_length = model_max_length


        self.video_data = self._preprocess_videos()
        self.prompt_ids = self._preprocess_prompts()

    def _preprocess_videos(self):
        """
        Load all videos, split and sample their frames with the configured policy, and keep the results in memory.
        """
        all_video_clips = []
        for video_path in self.video_paths:
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found at {video_path}")
                continue

            try:
                vr = decord.VideoReader(video_path, width=self.width, height=self.height)
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                continue

            total_frames = len(vr)
            for i in range(self.block):
                start = int(self.waste_time * self.fps + (self.clip_duration+self.waste_time) * self.fps * i)
                for j in range(self.clips):
                    clip_frame_length = int(2 * self.fps)
                    start_frame = start + clip_frame_length * j + 1
                    end_frame = start_frame + clip_frame_length

                    if start_frame >= total_frames:
                        break

                    end_frame = min(end_frame, total_frames)

                    clip = vr.get_batch([i for i in range(start_frame, end_frame)])
                    video = clip[::self.sample_frame_rate][:self.n_sample_frames]


                    clip_tensor = torch.from_numpy(video.numpy()).permute(0, 3, 1, 2)
                    clip_tensor = clip_tensor.float() / 255.0
                    clip_tensor = clip_tensor * 2.0 - 1.0

                    all_video_clips.append(clip_tensor)

        return all_video_clips

    def _preprocess_prompts(self):
        """
        Read the text file and encode its entries with the tokenizer.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer must be provided to preprocess prompts.")

        with open(self.prompt_path, 'r') as f:
            text_prompts = [line.strip() for line in f]

        return self.tokenizer(
            text_prompts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

    def __len__(self):

        return min(len(self.video_data), self.prompt_ids.shape[0])

    def __getitem__(self, index):
        example = {
            "pixel_values": self.video_data[index],
            "prompt_ids": self.prompt_ids[index],
        }
        return example