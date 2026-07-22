

























































































































































































import os
import cv2
import argparse
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Match query videos to GT videos via frame‑feature matching")
    parser.add_argument("--gt_root", type=str, default="/path/to/gt", help="Path to GT frames root (with subfolders each a video)")
    parser.add_argument("--query_root", type=str, default="/path/to/cine_showed_recons", help="Path to query frames root (with subfolders each a video)")
    parser.add_argument("--max_gt_frames", type=int, default=100, help="Max number of frames to sample per GT video")
    parser.add_argument("--stride_gt", type=int, default=3, help="Frame‑sampling stride for GT videos")
    parser.add_argument("--max_query_frames", type=int, default=50, help="Max number of frames to sample per query video")
    parser.add_argument("--stride_query", type=int, default=1, help="Frame‑sampling stride for query videos")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K matches for each query video")
    parser.add_argument("--distance_thresh", type=int, default=30, help="Distance threshold for ‘good’ feature matches")
    return parser.parse_args()

def extract_features(img_path, detector):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    kp, desc = detector.detectAndCompute(img, None)
    return desc

def load_frame_features_from_folder(folder_path, detector, stride):
    descs = []
    names = sorted(os.listdir(folder_path))
    for idx, fname in enumerate(names):
        if idx % stride != 0:
            continue
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        d = extract_features(fpath, detector)
        if d is not None:
            descs.append(d)
    return descs

def match_descs(q_descs, gt_descs_list, matcher, distance_thresh):
    total_good = 0
    for qd in q_descs:
        for gd in gt_descs_list:
            if qd is None or gd is None:
                continue
            matches = matcher.match(qd, gd)
            good = [m for m in matches if m.distance < distance_thresh]
            total_good += len(good)
    return total_good

def main():
    args = parse_args()
    detector = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print("Loading GT features …")
    gt_features = {}
    for name in tqdm(sorted(os.listdir(args.gt_root))):
        sub = os.path.join(args.gt_root, name)
        if not os.path.isdir(sub):
            continue
        descs = load_frame_features_from_folder(sub, detector, stride=args.stride_gt)
        if descs:
            gt_features[name] = descs
        else:
            print(f"Warning: no descriptors for GT folder {name}")
    print(f"Finished loading {len(gt_features)} GT video folders.\n")

    print("Matching query videos …")
    for qname in sorted(os.listdir(args.query_root)):
        qsub = os.path.join(args.query_root, qname)
        if not os.path.isdir(qsub):
            continue
        q_descs = load_frame_features_from_folder(qsub, detector, stride=args.stride_query)
        if not q_descs:
            print(f"Warning: no descriptors for query folder {qname}")
            continue

        scores = []
        for gt_name, gt_descs in gt_features.items():
            score = match_descs(q_descs, gt_descs, matcher, args.distance_thresh)
            scores.append((gt_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:args.top_k]

        print(f"Query folder '{qname}' → Top {args.top_k} GT matches:")
        for rank, (gt_name, score) in enumerate(top_k, start=1):
            print(f"  {rank}. GT folder '{gt_name}', score = {score}")
        print("")

if __name__ == "__main__":
    main()
