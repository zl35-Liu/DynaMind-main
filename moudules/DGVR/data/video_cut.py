import os
import cv2
import argparse
from pathlib import Path

def split_videos(source_path, target_path, segment_duration=2):
    """
    Split every video in the source directory into an initial segment and the remaining segment. The initial segment duration defaults to two seconds.
    """


    Path(target_path).mkdir(parents=True, exist_ok=True)


    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']


    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(source_path).glob(f'*{ext}'))
        video_files.extend(Path(source_path).glob(f'*{ext.upper()}'))

    if not video_files:
        print(f"No video files were found in  {source_path} ")
        return

    print(f"Found  {len(video_files)}  video files")

    for video_file in video_files:
        try:

            cap = cv2.VideoCapture(str(video_file))


            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps == 0:
                print(f"Warning: unable to obtain the frame rate for  {video_file.name} ; skipping the file")
                continue


            segment_frames = int(segment_duration * fps)

            if segment_frames >= total_frames:
                print(f"Warning: {video_file.name}  video duration is shorter than {segment_duration} seconds; skipping split")
                continue


            filename_stem = video_file.stem
            first_path = Path(target_path) / "first"
            second_path = Path(target_path) / "second"
            first_path.mkdir(parents=True, exist_ok=True)
            second_path.mkdir(parents=True, exist_ok=True)
            first_part_path = Path(first_path) / f"{filename_stem}{video_file.suffix}"
            second_part_path = Path(second_path) / f"{filename_stem}{video_file.suffix}"


            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_first = cv2.VideoWriter(str(first_part_path), fourcc, fps, (width, height))
            out_second = cv2.VideoWriter(str(second_part_path), fourcc, fps, (width, height))

            frame_count = 0

            print(f"Processing: {video_file.name}")

            while True:
                ret, frame = cap.read()

                if not ret:
                    break


                if frame_count < segment_frames:
                    out_first.write(frame)
                else:
                    out_second.write(frame)

                frame_count += 1


            cap.release()
            out_first.release()
            out_second.release()

            print(f"Completed: {filename_stem} -> split into two parts")

        except Exception as e:
            print(f"Processing  {video_file.name}  failed: {str(e)}")

    print("All videos have been processed.")

def main():
    parser = argparse.ArgumentParser(description='Split videos into the first two seconds and the remaining segment')
    parser.add_argument('source_path', help='Source video path')
    parser.add_argument('target_path', help='Output path')
    parser.add_argument('--duration', type=int, default=2,
                       help='Duration of the initial segment in seconds; default is two')

    args = parser.parse_args()


    if not os.path.exists(args.source_path):
        print(f"Error: source path  {args.source_path}  does not exist")
        return

    split_videos(args.source_path, args.target_path, args.duration)

if __name__ == "__main__":





    source_dir = '/path/to/cinebrain/dataset/clips'
    target_dir = '/path/to/cinebrain/dataset/clips_split'

    if source_dir and target_dir:
        split_videos(source_dir, target_dir)
    else:
        print("Paths cannot be empty")