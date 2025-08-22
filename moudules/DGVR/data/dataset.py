import os
import decord
# from triton.language import tensor

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
            n_sample_frames: int = 8,  # 从视频中采样的帧数
            sample_start_idx: int = 0,   # 采样的起始索引
            sample_frame_rate: int = 1,  # 采样的帧率   应该是间隔多少帧采一次？
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
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)    #decord 是一个高效的视频解码库，VideoReader 用于读取视频文件，指定了视频的宽度和高度（即帧的分辨率）
        # 根据设定的采样起始索引（sample_start_idx）、采样帧率（sample_frame_rate）和采样帧数（n_sample_frames），从视频中计算出需要采样的帧索引
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        # sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))
        print(len(sample_index))

        video = vr.get_batch(sample_index)  # 使用 decord 的 get_batch 方法加载指定索引的帧
        video = rearrange(video, "f h w c -> f c h w")  # 形状从 (frames, height, width, channels) 转换为 (frames, channels, height, width)

        # normalize to [-1, 1] 将视频数据标准化到 [-1, 1] 范围，并将文本提示（prompt_ids）与视频数据一起返回
        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example


# 修改多段视频加载
class TuneMultiVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: list,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 8,   # 应该设为8（每8帧采样一次，视频24帧） 这样就对上了最后生成视频的2s有6帧

            block: int = 40,        # 40个概念 每一块都是一个要舍弃的片头和余下的内容
            clips: int = 5,          # 每个概念5个视频
            waste_time: float = 3.0,  # 要舍弃的时间
            clip_duration: float = 10.0,  # 每个片段持续2秒
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
        # return 1
        return self.block * self.clips   # len 代表传入数据量

    def __getitem__(self, index):
        # 加载视频
        try:
            vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        except Exception as e:
            print(f"Error loading video {self.video_path}: {e}")
        # fps = vr.get_avg_fps()  # 获取视频帧率
        fps = 24  # 视频24帧
        total_frames = len(vr)  # 获取视频总帧数
        # print(total_frames)

        block_time=self.waste_time+self.clip_duration
        # sample_index = []  # 存储采样帧的索引 遍历所有块逐步添加 最后导入视频
        total=[]
        for i in range(self.block):
            # 计算每个片段的起始帧和结束帧
            start = int(self.waste_time * fps+block_time*fps*i)  # 当前块开始的帧索引

            for j in range(5):


                clip_frame_length = int(2 * fps )  # 每个2s的帧数
                start_frame = start + clip_frame_length*j+1 # 当前2s开始的帧索引
                end_frame = start_frame + clip_frame_length  # 当前片段的结束帧
                if end_frame >12480:
                    end_frame = 12480
                # if(i==39):
                    # print("第",i,"块的第",j,"个片段的起始帧index ",start_frame,"结束帧index ",end_frame)

                # 确保片段不超过视频总长度
                if end_frame > total_frames+1:
                    raise ValueError(f"Clip {index} exceeds video length.")

                # 获取每个片段的帧
                clip = vr.get_batch(range(start_frame, end_frame))
                # print("2s片段有多少帧",len(clip))  #检查正确 2s 48帧
                clip = rearrange(clip, "f h w c -> f c h w")  # 转换为 (frames, channels, height, width)

                # 采样选定帧
                # video = clip[::self.sample_frame_rate][:self.n_sample_frames]  # 每8帧采样一次，总共采样6帧
                video = clip[::self.sample_frame_rate]  # 每8帧采样一次，总共采样6帧


                # 标准化到 [-1, 1]
                example = {
                    "pixel_values": (video / 127.5 - 1.0),
                    "prompt_ids": self.prompt_ids[i*5+j]  # i块j个
                }
                # total.update(example)
                total.append(example)

        return total



# try实现multi 切2s
class TuneMultiVideoDataset1(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: list,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 8,   # 应该设为8（每8帧采样一次，视频24帧） 这样就对上了最后生成视频的2s有6帧

            block: int = 40,        # 40个概念 每一块都是一个要舍弃的片头和余下的内容
            clips: int = 5,          # 每个概念5个视频
            waste_time: float = 3.0,  # 要舍弃的时间
            clip_duration: float = 10.0,  # 每个片段持续2秒
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
        # return 1
        return self.block * self.clips   # len 代表传入数据

    def __getitem__(self, index):
        total = []
        subnames = os.listdir(self.video_path)
        for subname in subnames:
            subpath = os.path.join(self.video_path, subname)
            print(subpath)
            # 加载视频
            try:
                vr = decord.VideoReader(subpath, width=self.width, height=self.height)
            except Exception as e:
                print(f"Error loading video {self.video_path}: {e}")
            # fps = vr.get_avg_fps()  # 获取视频帧率
            fps = 24  # 视频24帧
            total_frames = len(vr)  # 获取视频总帧数
            # print(total_frames)

            block_time=self.waste_time+self.clip_duration
            # sample_index = []  # 存储采样帧的索引 遍历所有块逐步添加 最后导入视频

            for i in range(self.block):
                # 计算每个片段的起始帧和结束帧
                start = int(self.waste_time * fps+block_time*fps*i)  # 当前块开始的帧索引

                for j in range(5):


                    clip_frame_length = int(2 * fps )  # 每个2s的帧数
                    start_frame = start + clip_frame_length*j+1 # 当前2s开始的帧索引
                    end_frame = start_frame + clip_frame_length  # 当前片段的结束帧
                    if end_frame >12480:
                        end_frame = 12480
                    # if(i==39):
                        # print("第",i,"块的第",j,"个片段的起始帧index ",start_frame,"结束帧index ",end_frame)

                    # 确保片段不超过视频总长度
                    if end_frame > total_frames+1:
                        raise ValueError(f"Clip {index} exceeds video length.")

                    # 获取每个片段的帧
                    clip = vr.get_batch(range(start_frame, end_frame))
                    # print("2s片段有多少帧",len(clip))  #检查正确 2s 48帧
                    clip = rearrange(clip, "f h w c -> f c h w")  # 转换为 (frames, channels, height, width)

                    # 采样选定帧
                    # video = clip[::self.sample_frame_rate][:self.n_sample_frames]  # 每8帧采样一次，总共采样6帧
                    video = clip[::self.sample_frame_rate]  # 每8帧采样一次，总共采样6帧


                    # 标准化到 [-1, 1]
                    example = {
                        "pixel_values": (video / 127.5 - 1.0),
                        "prompt_ids": self.prompt_ids[i*5+j]  # i块j个
                    }
                    # total.update(example)
                    total.append(example)

        return total


# 外面切分 getitem只根据传入tensor索引
class TuneMultiVideoDataset2(Dataset):
    def __init__(
            self,
            video: list,
            prompt: list,
            width: int = 128,
            height: int = 72,
            n_sample_frames: int = 6,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 8,   # 应该设为8（每8帧采样一次，视频24帧） 这样就对上了最后生成视频的2s有6帧

            block: int = 40,        # 40个概念 每一块都是一个要舍弃的片头和余下的内容
            clips: int = 5,          # 每个概念5个视频
            waste_time: float = 3.0,  # 要舍弃的时间
            clip_duration: float = 10.0,  # 每个片段持续2秒
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
        # return 1
        return len(self.video)   # len 代表传入数据

    def __getitem__(self, index):

        # 标准化到 [-1, 1]
        example = {
            "pixel_values": (self.video[index] / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[index]  # i块j个
        }
        return example