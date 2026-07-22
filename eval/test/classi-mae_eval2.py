import os
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import imageio
import cv2
import warnings


try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, AutoImageProcessor, VideoMAEConfig

    _has_transformers = True
    print("Hugging Face Transformers was found. VideoMAE from transformers will be used for evaluation.")
except ImportError:
    _has_transformers = False
    print("Error: Hugging Face Transformers was not found. Install it with: pip install transformers imageio[ffmpeg]")
    exit()



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



warnings.filterwarnings("ignore",
                        message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.")
warnings.filterwarnings("ignore",
                        message="The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.")




GIF_DIR = "/path/to/workspace/Disk/hdd-1/Ljx/EEG2Video-main/EEG2Video/40_Classes_EEG25"

block_num = 1
pth_num = 1


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
for block_id in range(block_num,block_num+1):
    All_label = np.concatenate((All_label, GT_label[block_id].repeat(5).reshape(1, 200)))
LABELS = rearrange(All_label, 'b c -> (b c)') - 1

NUM_CLASSES_40 = 40
BATCH_SIZE = 16



MODEL_STANDARD_IMAGE_SIZE = 224

NUM_FRAMES_VIDEOMAE = 6

VIDEO_MAE_PATCH_SIZE = (2, 16, 16)


LOCAL_MODEL_PATH = "./videoMAE"

PRE_MODEL_PATH = f"./checkpoints/best_40_class_videomae-{pth_num}.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print(f"\nLoading the VideoMAE model and ImageProcessor from  {LOCAL_MODEL_PATH}...")



processor_config_obj = VideoMAEImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
processor_config_obj.do_resize = True
processor_config_obj.size = {"shortest_edge": MODEL_STANDARD_IMAGE_SIZE}
processor_config_obj.do_center_crop = True
processor_config_obj.crop_size = {"height": MODEL_STANDARD_IMAGE_SIZE, "width": MODEL_STANDARD_IMAGE_SIZE}
processor_config_obj.num_frames = NUM_FRAMES_VIDEOMAE

image_processor = VideoMAEImageProcessor(**processor_config_obj.to_dict())
print("VideoMAE ImageProcessor was loaded and configured.")



model_config = VideoMAEConfig.from_pretrained(LOCAL_MODEL_PATH)


model_config.num_frames = NUM_FRAMES_VIDEOMAE
model_config.image_size = MODEL_STANDARD_IMAGE_SIZE
model_config.patch_size = (VIDEO_MAE_PATCH_SIZE[1], VIDEO_MAE_PATCH_SIZE[2])
model_config.tube_patch_size = VIDEO_MAE_PATCH_SIZE
model_config.num_labels = NUM_CLASSES_40


model_videomae = VideoMAEForVideoClassification(model_config)
print(f"Loading from  {PRE_MODEL_PATH} Loading fine-tuned model weights...")



model_videomae.load_state_dict(torch.load(PRE_MODEL_PATH, map_location='cpu'))

model_videomae = model_videomae.to(device)
model_videomae.eval()
print("The fine-tuned VideoMAE model was loaded and set to evaluation mode.")



def preprocess_gif_for_videomae(gif_path, processor, num_frames_to_sample):
    """
    Convert a GIF to a sampled frame sequence and process it with a configured VideoMAE image processor.
    """
    gif = imageio.mimread(gif_path)
    if not gif:
        print(f"Warning: unable to read the GIF or it is empty: {gif_path}")

        return torch.zeros((3, num_frames_to_sample, processor.size["height"], processor.size["width"])).float()

    video_frames_np = []
    for frame in gif:

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            pass
        else:
            print(f"Warning: uncommon GIF frame format: {frame.shape}, skipping frame.")
            continue
        video_frames_np.append(frame)

    video_frames_np = np.array(video_frames_np)

    total_frames_in_gif = video_frames_np.shape[0]
    if total_frames_in_gif == 0:
        print(f"Warning: GIF  {gif_path}  contains no valid frames.")
        return torch.zeros((3, num_frames_to_sample, processor.size["height"], processor.size["width"])).float()


    if total_frames_in_gif == num_frames_to_sample:
        sampled_frames = video_frames_np
    elif total_frames_in_gif < num_frames_to_sample:

        indices = np.random.choice(total_frames_in_gif, num_frames_to_sample, replace=True)
        sampled_frames = video_frames_np[indices]
    else:

        indices = np.linspace(0, total_frames_in_gif - 1, num_frames_to_sample).astype(int)
        sampled_frames = video_frames_np[indices]





    inputs = processor(list(sampled_frames), return_tensors="pt").pixel_values.squeeze(0)




    return inputs



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
            pred_picked_sorted_indices = pred_picked.argsort(descending=True)

            if 0 in pred_picked_sorted_indices[:top_k]:
                corrects += 1
                break

    if num_trials == 0:
        return 0.0, 0.0

    accuracy = corrects / num_trials
    std = np.sqrt(accuracy * (1 - accuracy) / num_trials) if num_trials > 0 else 0.0
    return accuracy, std



def evaluate_video_quality_with_videomae():
    """
    Evaluate generated-video semantics with VideoMAE and compute N-way top-k accuracy.
    """
    all_labels_40 = []
    all_probs_40 = []


    gif_files = sorted([f for f in os.listdir(GIF_DIR) if f.endswith(".gif")])
    gif_files = sorted(gif_files, key=extract_number_from_filename)
    gif_paths = [os.path.join(GIF_DIR, f) for f in gif_files]
    print(f"Found  {len(gif_paths)}  GIF files for evaluation.")


    assert len(gif_paths) == len(LABELS), f"GIF file count ({len(gif_paths)}) and label count ({len(LABELS)}) do not match"

    print(f"Starting evaluation of  {len(gif_paths)}  videos...")


    for i in range(0, len(gif_paths), BATCH_SIZE):
        batch_paths = gif_paths[i:i + BATCH_SIZE]
        batch_labels_40 = LABELS[i:i + BATCH_SIZE]
        batch_video_inputs = []

        print(f"Processing batch  {i // BATCH_SIZE + 1}/{(len(gif_paths) - 1) // BATCH_SIZE + 1}...")


        for path in batch_paths:
            video_input_tensor = preprocess_gif_for_videomae(
                path,
                image_processor,
                NUM_FRAMES_VIDEOMAE
            )
            batch_video_inputs.append(video_input_tensor)


        video_batch = torch.stack(batch_video_inputs).to(device)


        with torch.no_grad():
            outputs_40 = model_videomae(pixel_values=video_batch).logits
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
        f"Mean N-way Top-{N_WAY_K_ACC_TOP_K}  accuracy (N={N_WAY_K_ACC_N2}): {mean_n_way_top_k_acc2:.4f} ± {mean_n_way_top_k_std2:.4f}")

    return mean_n_way_top_k_acc, mean_n_way_top_k_std



if __name__ == "__main__":

    mean_n_way_top_k_acc, mean_n_way_top_k_std = evaluate_video_quality_with_videomae()


    save_path = "./evaluation_results_videomae_n_way_top_k.txt"
    with open(save_path, "w") as f:
        f.write(f"VideoMAE mean N-way Top-1 accuracy (N=40): {mean_n_way_top_k_acc:.4f}\n")
        f.write(f"VideoMAE mean N-way Top-1 standard deviation (N=40): {mean_n_way_top_k_std:.4f}\n")

    print(f"Evaluation completed. Results saved to  {save_path}")