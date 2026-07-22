import numpy as np
import os
from einops import rearrange


def preprocess_and_save_labels():
    """
    Preprocess and save labels for every classification task using the configured paths.
    """

    DATA_DIR = "/path/to/DynaMind-main/"
    META_INFO_DIR = "data/Video/meta-info"
    SAMPLES_PER_BLOCK = 200
    NUM_BLOCKS = 7

    meta_info_path = os.path.join(DATA_DIR, META_INFO_DIR)
    os.makedirs(meta_info_path, exist_ok=True)

    print(f"Loading and processing labels for saving to {meta_info_path}...")



    GT_label_raw = np.array([[23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33,
                              34, 4, 39, 17, 1, 26, 20, 29, 13, 32, 37, 2, 11, 12, 30, 31, 8, 21, 7, 10, ],
                             [27, 33, 22, 28, 31, 12, 38, 4, 18, 17, 35, 39, 40, 5, 24, 32, 15, 13, 2, 16,
                              34, 25, 19, 30, 23, 3, 8, 29, 7, 20, 11, 14, 37, 6, 21, 1, 10, 36, 26, 9, ],
                             [15, 36, 31, 1, 34, 3, 37, 12, 4, 5, 21, 24, 14, 16, 39, 20, 28, 29, 18, 32,
                              2, 27, 8, 19, 13, 10, 30, 40, 17, 26, 11, 9, 33, 25, 35, 7, 38, 22, 23, 6, ],
                             [16, 28, 23, 1, 39, 10, 35, 14, 19, 27, 37, 31, 5, 18, 11, 25, 29, 13, 20, 24,
                              7, 34, 26, 4, 40, 12, 8, 22, 21, 30, 17, 2, 38, 9, 3, 36, 33, 6, 32, 15, ],
                             [18, 29, 7, 35, 22, 19, 12, 36, 8, 15, 28, 1, 34, 23, 20, 13, 37, 9, 16, 30,
                              2, 33, 27, 21, 14, 38, 10, 17, 31, 3, 24, 39, 11, 32, 4, 25, 40, 5, 26, 6, ],
                             [29, 16, 1, 22, 34, 39, 24, 10, 8, 35, 27, 31, 23, 17, 2, 15, 25, 40, 3, 36,
                              26, 6, 14, 37, 9, 12, 19, 30, 5, 28, 32, 4, 13, 18, 21, 20, 7, 11, 33, 38],
                             [38, 34, 40, 10, 28, 7, 1, 37, 22, 9, 16, 5, 12, 36, 20, 30, 6, 15, 35, 2,
                              31, 26, 18, 24, 8, 3, 23, 19, 14, 13, 21, 4, 25, 11, 32, 17, 39, 29, 33, 27]
                             ])
    all_class_labels = np.empty((0, SAMPLES_PER_BLOCK))
    for block_id in range(NUM_BLOCKS):
        all_class_labels = np.concatenate((all_class_labels, GT_label_raw[block_id].repeat(5).reshape(1, 200)))
    processed_video_category = rearrange(all_class_labels, 'b c -> (b c)') - 1

    np.save(os.path.join(meta_info_path, "All_video_category_40.npy"), processed_video_category)
    print(f"Processed 'video_category' labels saved to {os.path.join(meta_info_path, 'All_video_category.npy')}")



    GT_label6_raw = np.copy(GT_label_raw)
    for i in range(NUM_BLOCKS):
        for j in range(40):
            if 1 <= GT_label6_raw[i][j] <= 7:
                GT_label6_raw[i][j] = 0
            elif 8 <= GT_label6_raw[i][j] <= 11:
                GT_label6_raw[i][j] = 1
            elif 12 <= GT_label6_raw[i][j] <= 14:
                GT_label6_raw[i][j] = 2
            elif 15 <= GT_label6_raw[i][j] <= 18:
                GT_label6_raw[i][j] = 3
            elif 19 <= GT_label6_raw[i][j] <= 21:
                GT_label6_raw[i][j] = 4
            elif 22 <= GT_label6_raw[i][j] <= 27:
                GT_label6_raw[i][j] = 5
            elif 28 <= GT_label6_raw[i][j] <= 32:
                GT_label6_raw[i][j] = 6
            elif 33 <= GT_label6_raw[i][j] <= 36:
                GT_label6_raw[i][j] = 7
            else:
                GT_label6_raw[i][j] = 8

    all_GT_label6 = np.empty((0, SAMPLES_PER_BLOCK))
    for block_id in range(NUM_BLOCKS):
        all_GT_label6 = np.concatenate((all_GT_label6, GT_label6_raw[block_id].repeat(5).reshape(1, 200)))
    processed_concept = rearrange(all_GT_label6, 'b c -> (b c)')

    np.save(os.path.join(meta_info_path, "All_video_category_9.npy"), processed_concept)
    print(f"Processed 'concept' labels saved to {os.path.join(meta_info_path, 'All_video_concept_binning.npy')}")

    print("Labels preprocessing finished.")


if __name__ == "__main__":


    preprocess_and_save_labels()