import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import timm
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")



def setup_device():
    """
    Select an available GPU or fall back to the CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return torch.device(device)

def load_models(device):
    """
    Load the CLIP and DINOv2 models together with their preprocessing pipelines.
    """
    print("Loading CLIP model...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()

    print("Loading DINOv2 model...")

    dino_model_name = "dinov2_vits14"
    dino_model = timm.create_model(dino_model_name, pretrained=True).to(device).eval()


    dino_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("Models loaded.")
    return clip_model, clip_processor, dino_model, dino_transform



def extract_features(video_path, model, processor, model_type, device, batch_size=32):
    """
    Extract frame features from a video in batches with either CLIP or DINO using the supplied model, processor, device, and batch size.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return torch.empty(0)

    frames_pil = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames_pil:
        print(f"No frames read from video: {video_path}")
        return torch.empty(0)

    all_features = []
    with torch.no_grad():
        for i in range(0, len(frames_pil), batch_size):
            batch_pil = frames_pil[i : i + batch_size]

            try:
                if model_type == 'clip':
                    inputs = processor(images=batch_pil, return_tensors="pt", padding=True).to(device)
                    features = model.get_image_features(**inputs)

                elif model_type == 'dino':

                    batch_tensor = torch.stack([processor(img) for img in batch_pil]).to(device)

                    features = model.forward_features(batch_tensor)[:, 0]


                features = F.normalize(features, p=2, dim=1)
                all_features.append(features.cpu())

            except Exception as e:
                print(f"Error processing batch for {video_path}: {e}")
                continue

    if not all_features:
        return torch.empty(0)

    return torch.cat(all_features, dim=0)

def calculate_temporal_consistency(features_tensor):
    """
    Compute the mean cosine similarity between consecutive frame features in a tensor shaped (frames, feature dimension).
    """
    if features_tensor.shape[0] < 2:
        return 0.0


    features_1 = features_tensor[:-1]

    features_2 = features_tensor[1:]


    similarities = F.cosine_similarity(features_1, features_2, dim=1)


    return similarities.mean().item()



def process_directory(video_paths, clip_model, clip_processor, dino_model, dino_transform, device):
    """
    Process every video in a directory and calculate the mean evaluation metrics.
    """
    ctc_scores = []
    dtc_scores = []

    if not video_paths:
        print("No videos found in directory.")
        return 0.0, 0.0

    for video_path in tqdm(video_paths, desc="Processing videos"):

        clip_features = extract_features(video_path, clip_model, clip_processor, 'clip', device)
        ctc_score = calculate_temporal_consistency(clip_features)
        ctc_scores.append(ctc_score)


        dino_features = extract_features(video_path, dino_model, dino_transform, 'dino', device)
        dtc_score = calculate_temporal_consistency(dino_features)
        dtc_scores.append(dtc_score)


    avg_ctc = np.mean(ctc_scores) if ctc_scores else 0.0
    avg_dtc = np.mean(dtc_scores) if dtc_scores else 0.0

    return avg_ctc, avg_dtc

def get_video_files(directory):
    """
    Return a sorted list of MP4 and AVI files in a directory.
    """
    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    p = Path(directory)
    video_files = [
        f for f in p.glob("*")
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    return sorted(video_files)

def main():


    PATH_GT = Path("./path/to/your/ground_truth_videos")

    PATH_GEN = Path("./path/to/your/generated_videos")


    device = setup_device()
    clip_model, clip_processor, dino_model, dino_transform = load_models(device)

    print(f"\nProcessing Ground Truth videos from: {PATH_GT}")
    gt_video_files = get_video_files(PATH_GT)
    if not gt_video_files:
        print(f"Warning: No video files found in {PATH_GT}")
        avg_gt_ctc, avg_gt_dtc = 0.0, 0.0
    else:
        avg_gt_ctc, avg_gt_dtc = process_directory(
            gt_video_files, clip_model, clip_processor, dino_model, dino_transform, device
        )

    print(f"\nProcessing Generated videos from: {PATH_GEN}")
    gen_video_files = get_video_files(PATH_GEN)
    if not gen_video_files:
        print(f"Warning: No video files found in {PATH_GEN}")
        avg_gen_ctc, avg_gen_dtc = 0.0, 0.0
    else:
        avg_gen_ctc, avg_gen_dtc = process_directory(
            gen_video_files, clip_model, clip_processor, dino_model, dino_transform, device
        )


    if len(gt_video_files) != len(gen_video_files):
        print("\n---")
        print(f"Warning: Video counts do not match!")
        print(f"Ground Truth videos: {len(gt_video_files)}")
        print(f"Generated videos: {len(gen_video_files)}")
        print("The script assumes matching video counts and order for a 1:1 comparison.")
        print("---")


    print("\n--- FINAL RESULTS ---")
    print(f"Path: {PATH_GT}")
    print(f"  Average Ground Truth CTC: {avg_gt_ctc:.4f}")
    print(f"  Average Ground Truth DTC: {avg_gt_dtc:.4f}")
    print(f"\nPath: {PATH_GEN}")
    print(f"  Average Generated CTC: {avg_gen_ctc:.4f}")
    print(f"  Average Generated DTC: {avg_gen_dtc:.4f}")
    print("---------------------")

if __name__ == "__main__":
    main()