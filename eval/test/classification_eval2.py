import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import imageio
import cv2
import pickle
import matplotlib.pyplot as plt







GIF_DIR = "/path/to/workspace/Disk/hdd-1/Ljx/EEG2Video-main/EEG2Video/reconstruction/40/sub1_session2/0-1"
video_block = 0
pth_num = 2
GT_label = np.array([[23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33,
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
All_label = np.empty((0, 200))
for block_id in range(video_block, video_block + 1):
    All_label = np.concatenate((All_label, GT_label[block_id].repeat(5).reshape(1, 200)))
LABELS = rearrange(All_label, 'b c -> (b c)') - 1

NUM_CLASSES_40 = 40
BATCH_SIZE = 32
TARGET_SIZE = (288, 512)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_40 = torchvision.models.resnet50()
num_ftrs = model_40.fc.in_features
model_40.fc = nn.Linear(num_ftrs, NUM_CLASSES_40)
model_40.load_state_dict(torch.load(f"./checkpoints/best_resnet_40_class-{pth_num}.pth"))
model_40 = model_40.to(device)
model_40.eval()
print("The 40-class model was loaded.")



def preprocess_gif(gif_path):
    """
    Convert a GIF into a single image frame.
    """
    gif = imageio.mimread(gif_path)
    mid_idx = len(gif) // 2
    frame = gif[mid_idx]

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    elif len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    frame = cv2.resize(frame, TARGET_SIZE)

    if frame.max() > 1.0:
        frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std

    frame = frame.transpose(2, 0, 1)
    return frame


def extract_number_from_filename(filename):
    """
    Extract the numeric identifier from a filename such as 100.gif.
    """
    name_without_extension = os.path.splitext(filename)[0]
    try:
        return int(name_without_extension)
    except ValueError:
        print(f"Warning: filename '{filename}' cannot be parsed as a number, which may affect sorting.")
        return float('inf')



gif_files = sorted([f for f in os.listdir(GIF_DIR) if f.endswith(".gif")])
gif_files = sorted(gif_files, key=extract_number_from_filename)
gif_paths = [os.path.join(GIF_DIR, f) for f in gif_files]
print(gif_paths)


assert len(gif_paths) == len(LABELS), f"File count ({len(gif_paths)}) and label count ({len(LABELS)}) do not match"



def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):


    if not isinstance(class_id, list):
        class_id = [int(class_id)]


    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if pred.dim() > 1:
        pred = pred.flatten()

    pick_range = [i for i in np.arange(len(pred)) if i not in class_id]


    replace_val = False
    if n_way - 1 > len(pick_range):
        print(f"Warning: selected negative count ({n_way - 1}) exceeds the available negative count ({len(pick_range)})。")
        print(f"For class ID  {class_id}, available prediction range:  {len(pred)}, negative sample pool:  {len(pick_range)}")
        if len(pick_range) == 0 and n_way - 1 > 0:
            print("No negative samples can be selected; this trial may be invalid or skipped.")
            return 0.0, 0.0

        replace_val = True

    corrects = 0
    for t in range(num_trials):

        if replace_val and len(pick_range) == 0 and n_way - 1 > 0:
            continue

        idxs_picked = np.random.choice(pick_range, n_way - 1, replace=replace_val)

        for gt_id in class_id:

            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])



            pred_picked = pred_picked.argsort(descending=False)[-top_k:]

            if 0 in pred_picked:
                corrects += 1
                break


    if num_trials == 0:
        return 0.0, 0.0

    accuracy = corrects / num_trials

    std = np.sqrt(accuracy * (1 - accuracy) / num_trials) if num_trials > 0 else 0.0
    return accuracy, std



def evaluate_video_quality():
    """
    Evaluate the semantic quality of generated videos and compute N-way top-k accuracy.
    """
    all_labels_40 = []
    all_probs_40 = []

    print(f"Starting evaluation of  {len(gif_paths)}  videos...")


    for i in range(0, len(gif_paths), BATCH_SIZE):
        batch_paths = gif_paths[i:i + BATCH_SIZE]
        batch_labels_40 = LABELS[i:i + BATCH_SIZE]
        batch_frames = []

        print(f"Processing batch  {i // BATCH_SIZE + 1}/{(len(gif_paths) - 1) // BATCH_SIZE + 1}...")


        for path in batch_paths:
            frame = preprocess_gif(path)
            frame_tensor = torch.tensor(frame).float()
            batch_frames.append(frame_tensor)


        frame_batch = torch.stack(batch_frames).to(device)


        with torch.no_grad():
            outputs_40 = model_40(frame_batch)
            probs_40 = F.softmax(outputs_40, dim=1)


        all_probs_40.extend(probs_40.cpu().numpy())
        all_labels_40.extend(batch_labels_40)


    all_probs_40_np = np.array(all_probs_40)
    all_labels_40_np = np.array(all_labels_40)


    all_n_way_top_k_accs = []
    all_n_way_top_k_stds = []
    all_n_way_top_k_accs2 = []
    all_n_way_top_k_stds2 = []




    N_WAY_K_ACC_N = 40
    N_WAY_K_ACC_N2 = 2
    N_WAY_K_ACC_TRIALS = 40
    N_WAY_K_ACC_TOP_K = 1


    if N_WAY_K_ACC_N > NUM_CLASSES_40:
        print(f"Error: N_WAY_K_ACC_N ({N_WAY_K_ACC_N}) cannot exceed the total class count ({NUM_CLASSES_40}). Adjust N_WAY_K_ACC_N.")
        return 0.0, 0.0

    print(
        f"\nStarting per-sample N-way Top-{N_WAY_K_ACC_TOP_K}  accuracy (N={N_WAY_K_ACC_N}, trials={N_WAY_K_ACC_TRIALS})...")
    for i in range(len(all_labels_40_np)):
        sample_probs = all_probs_40_np[i]
        sample_label = all_labels_40_np[i]


        acc, std = n_way_top_k_acc(
            torch.from_numpy(sample_probs).float(),
            sample_label,
            n_way=N_WAY_K_ACC_N,
            num_trials=N_WAY_K_ACC_TRIALS,
            top_k=N_WAY_K_ACC_TOP_K
        )
        acc2, std2 = n_way_top_k_acc(
            torch.from_numpy(sample_probs).float(),
            sample_label,
            n_way=N_WAY_K_ACC_N2,
            num_trials=N_WAY_K_ACC_TRIALS,
            top_k=N_WAY_K_ACC_TOP_K
        )
        all_n_way_top_k_accs.append(acc)
        all_n_way_top_k_stds.append(std)
        all_n_way_top_k_accs2.append(acc2)
        all_n_way_top_k_stds2.append(std2)


    mean_n_way_top_k_acc = np.mean(all_n_way_top_k_accs) if all_n_way_top_k_accs else 0.0
    mean_n_way_top_k_std = np.mean(all_n_way_top_k_stds) if all_n_way_top_k_stds else 0.0
    mean_n_way_top_k_acc2 = np.mean(all_n_way_top_k_accs2) if all_n_way_top_k_accs2 else 0.0
    mean_n_way_top_k_std2 = np.mean(all_n_way_top_k_stds2) if all_n_way_top_k_stds2 else 0.0

    print(
        f"\nMean N-way Top-{N_WAY_K_ACC_TOP_K}  accuracy (N={N_WAY_K_ACC_N}): {mean_n_way_top_k_acc:.4f} ± {mean_n_way_top_k_std:.4f}")
    print(
        f"\nMean N-way Top-{N_WAY_K_ACC_TOP_K}  accuracy (N={N_WAY_K_ACC_N2}): {mean_n_way_top_k_acc2:.4f} ± {mean_n_way_top_k_std2:.4f}")

    return mean_n_way_top_k_acc, mean_n_way_top_k_std



if __name__ == "__main__":

    mean_n_way_top_k_acc, mean_n_way_top_k_std = evaluate_video_quality()


    save_path = "./evaluation_results_n_way_top_k.txt"




    print(f"Evaluation completed. Results saved to  {save_path}")