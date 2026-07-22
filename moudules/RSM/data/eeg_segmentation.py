import numpy as np
import os

from accelerate.test_utils.scripts.test_script import print_on
from tqdm import tqdm





fre = 200


def analyze_npy_file(file_path):

    data = np.load(file_path)


    print(f"Data shape: {data.shape}")


    for idx, dim_size in enumerate(data.shape):
        print(f"Dimension  {idx + 1}  size: {dim_size}")




def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".npy"):
                files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("/path/to/DynaMind-main/data/EEG")

print(sub_list)


for subname in sub_list:
    npydata = np.load('/path/to/DynaMind-main/data/EEG/' + subname)


    save_data = np.empty((0, 40, 5, 62, 2*fre))

    for block_id in range(7):
        print("block: ", block_id)
        now_data = npydata[block_id]
        print("now_data_shape:", now_data.shape)
        l = 0
        block_data = np.empty((0, 5, 62, 2*fre))
        for class_id in tqdm(range(40)):
            l += (3 * fre)
            class_data = np.empty((0, 62, 2*fre))
            for i in range(5):
                class_data = np.concatenate((class_data, now_data[:, l : l + 2*fre].reshape(1, 62, 2*fre)))
                l += (2 * fre)
            block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2*fre)))
        save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2*fre)))


    save_dir = "/path/to/DynaMind-main/data/EEG_processed/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + subname.split('.',1)[0], save_data)