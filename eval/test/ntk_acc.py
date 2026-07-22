import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import random


warnings.filterwarnings("ignore", category=UserWarning)



N_WAY = 5
TOP_K = 1

def setup_device():
    """
    Select an available GPU or fall back to the CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return torch.device(device)

def load_clip_model(device):
    """
    Load the CLIP model and its preprocessing pipeline.
    """
    print("Loading CLIP model for N-way evaluation...")
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    print("CLIP model loaded.")
    return model, processor



def extract_features_from_video(video_path, model, processor, device):
    """
    Extract L2-normalized CLIP features for every frame in a video and return a tensor shaped (frames, feature dimension).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frames_pil = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames_pil:
        return None

    all_features = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(frames_pil), batch_size):
            batch_pil = frames_pil[i : i + batch_size]

            inputs = processor(images=batch_pil, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)


            features = F.normalize(features.float(), p=2, dim=1)
            all_features.append(features.cpu())

    if not all_features:
        return None

    return torch.cat(all_features, dim=0)

def load_all_video_features(video_paths, model, processor, device):
    """
    Load features for all videos.
    """
    all_features = {}
    for video_path in tqdm(video_paths, desc="Extracting features"):
        features = extract_features_from_video(video_path, model, processor, device)
        if features is not None:
            all_features[video_path.name] = features
    return all_features



def calculate_nway_topk(features_gt, features_candidates, K):
    """
    Compute N-way top-k accuracy for a ground-truth feature and N candidate features containing the ground truth and distractors.
    """

    if features_gt.dim() == 1:
        features_gt = features_gt.unsqueeze(0)



    similarities = F.cosine_similarity(features_gt, features_candidates, dim=1)



    sorted_indices = torch.argsort(similarities, descending=True)





















    is_top_k = (0 in sorted_indices[:K])

    return 1.0 if is_top_k else 0.0


def run_nway_evaluation(gt_features_map, gen_features_map, N, K):
    """
    Run N-way top-k evaluation and calculate video-level and frame-level accuracy.
    """
    gt_names = sorted(gt_features_map.keys())
    gen_names = sorted(gen_features_map.keys())

    if len(gt_names) != len(gen_names):
        raise ValueError("The ground-truth and generated video counts do not match.")

    M = len(gt_names)


    all_gt_indices = list(range(M))

    video_accuracies = []
    frame_accuracies = []
    total_frame_count = 0

    print(f"\nRunning {N}-way top-{K} evaluation...")
    for i in tqdm(range(M)):
        gt_name = gt_names[i]
        gen_name = gen_names[i]

        F_GT_frames = gt_features_map[gt_name]
        F_GEN_frames = gen_features_map.get(gen_name)

        if F_GEN_frames is None:
            print(f"Warning: Missing generated video features for {gen_name}")
            continue

        num_frames = min(F_GT_frames.shape[0], F_GEN_frames.shape[0])
        total_frame_count += num_frames




        distractor_indices = all_gt_indices[:]
        distractor_indices.remove(i)


        if len(distractor_indices) < N - 2:
            print(f"Warning: Not enough distractors (only {len(distractor_indices)} available). Skipping this video.")
            continue

        distractor_indices = random.sample(distractor_indices, N - 2)


        F_DIST_frames_list = [
            gt_features_map[gt_names[d_idx]][:num_frames] for d_idx in distractor_indices
        ]





        F_GT_video = F_GT_frames[:num_frames].mean(dim=0)
        F_GEN_video = F_GEN_frames[:num_frames].mean(dim=0)
        F_DIST_video_list = [f.mean(dim=0) for f in F_DIST_frames_list]


        candidates_video = torch.stack([F_GEN_video] + F_DIST_video_list)

        video_acc = calculate_nway_topk(F_GT_video, candidates_video, K)
        video_accuracies.append(video_acc)




        frame_acc_sum = 0


        for t in range(num_frames):
            F_GT_frame_t = F_GT_frames[t]
            F_GEN_frame_t = F_GEN_frames[t]


            F_DIST_frame_t_list = [f[t] for f in F_DIST_frames_list]


            candidates_frame = torch.stack([F_GEN_frame_t] + F_DIST_frame_t_list)

            frame_acc_sum += calculate_nway_topk(F_GT_frame_t, candidates_frame, K)

        frame_accuracies.append(frame_acc_sum / num_frames)


    avg_video_acc = np.mean(video_accuracies) if video_accuracies else 0.0

    avg_frame_acc = np.mean(frame_accuracies) if frame_accuracies else 0.0

    return avg_video_acc, avg_frame_acc


def get_video_files(directory):
    """
    Return a sorted list of MP4 and AVI files in a directory.
    """
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    p = Path(directory)
    video_files = [
        f for f in p.glob("*")
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    return sorted(video_files)

def main():


    PATH_GT = Path("./path/to/your/ground_truth_videos")

    PATH_GEN = Path("./path/to/your/generated_videos")


    print(f"--- N-way Top-K Accuracy ({N_WAY}-way Top-{TOP_K}) ---")
    print(f"Feature Extractor: CLIP-ViT-B/32")

    device = setup_device()
    clip_model, clip_processor = load_clip_model(device)

    gt_video_files = get_video_files(PATH_GT)
    gen_video_files = get_video_files(PATH_GEN)

    if len(gt_video_files) == 0 or len(gen_video_files) == 0:
        print("Error: One or both directories are empty or contain no recognizable video files.")
        return

    if len(gt_video_files) != len(gen_video_files):
        print("\n--- WARNING ---")
        print(f"Ground Truth videos: {len(gt_video_files)}")
        print(f"Generated videos: {len(gen_video_files)}")
        print("The video counts do not match, which may make N-way evaluation inaccurate. Ensure both directories contain the same number of videos in matching filename order.")
        print("-----------------")


    print(f"\n--- Loading Ground Truth Features ({len(gt_video_files)} videos) ---")
    gt_features_map = load_all_video_features(gt_video_files, clip_model, clip_processor, device)
    print(f"\n--- Loading Generated Features ({len(gen_video_files)} videos) ---")
    gen_features_map = load_all_video_features(gen_video_files, clip_model, clip_processor, device)


    try:
        avg_video_acc, avg_frame_acc = run_nway_evaluation(
            gt_features_map, gen_features_map, N_WAY, TOP_K
        )


        print("\n--- FINAL N-WAY ACCURACY RESULTS ---")
        print(f"N-way: {N_WAY}, K: {TOP_K}")
        print(f"1. Video-level Accuracy: {avg_video_acc:.4f} ({avg_video_acc * 100:.2f}%)")
        print(f"2. Frame-level Accuracy: {avg_frame_acc:.4f} ({avg_frame_acc * 100:.2f}%)")
        print("------------------------------------")

    except ValueError as e:
        print(f"Error during evaluation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()