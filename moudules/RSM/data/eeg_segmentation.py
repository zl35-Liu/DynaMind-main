import numpy as np
import os

from accelerate.test_utils.scripts.test_script import print_on
from tqdm import tqdm  #进度条库

# segment a EEG data numpy array with the shape of (7 * 62 * 520s*fre) into 2-sec EEG segments
# segment it into a new array with the shape of (7 * 40 * 5 * 62 * 2s*fre),
# meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.

fre = 200


def analyze_npy_file(file_path):
    # 加载.npy文件
    data = np.load(file_path)

    # 输出数据的维度大小
    print(f"数据的形状为: {data.shape}")

    # 输出每个维度的大小
    for idx, dim_size in enumerate(data.shape):
        print(f"维度 {idx + 1} 的大小为: {dim_size}")



#遍历指定目录及其子目录，获取所有文件的文件名，并将它们存储到 files_names 列表中
def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):  #os.walk()遍历指定目录及其所有子目录，并返回一个三元组 (dirpath, dirnames, filenames)
        for filename in filenames:
            if filename.endswith(".npy"):
                files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("E:/store/DynaMind-main/data/EEG")   #sub_list 保存了目标目录中所有 .npy 文件的文件名

print(sub_list)


for subname in sub_list:
    npydata = np.load('E:/store/DynaMind-main/data/EEG/' + subname)
    # print(npydata)

    save_data = np.empty((0, 40, 5, 62, 2*fre))  #初始化存储分段数据的空数组(0, 40, 5, 62, 2*fre),0对应初始空

    for block_id in range(7):
        print("block: ", block_id)
        now_data = npydata[block_id]
        print("now_data_shape:", now_data.shape)
        l = 0   #初始化指针 l，用于在数据中切片
        block_data = np.empty((0, 5, 62, 2*fre))  #为当前块初始化一个空数组，形状为 (0, 5, 62, 2*fre)，用于存储当前块的数据
        for class_id in tqdm(range(40)):
            l += (3 * fre)  #hint跳过
            class_data = np.empty((0, 62, 2*fre))  #存储该类别的5个视频片段的EEG数据
            for i in range(5):
                class_data = np.concatenate((class_data, now_data[:, l : l + 2*fre].reshape(1, 62, 2*fre))) #now_data[:, l : l + 2*fre] 提取出一个2秒钟（400个时间点）的EEG数据，追加到 class_data 中
                l += (2 * fre)
            block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2*fre)))  #处理完一个类别的5个视频片段后，将 class_data（5个片段）追加到 block_data
        save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2*fre)))    #当前数据块（block_data）添加到 save_data
        #np.concatenate 拼接数组

    save_dir = "E:/store/DynaMind-main/data/EEG_processed/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + subname.split('.',1)[0], save_data)