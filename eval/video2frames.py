import cv2
import os
import glob

def extract_frames_from_folder(source_path):
    """
    Decode every video in the specified directory and save its frames in a subdirectory named after the video.

    Args:
        source_path: Directory containing the video files.
    """


    video_extensions = ('*.mp4', '*.gif', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv')


    video_files = []
    for ext in video_extensions:


        video_files.extend(glob.glob(os.path.join(source_path, ext)))

    if not video_files:
        print(f"No supported video files were found in '{source_path}'.")
        return

    print(f"Found  {len(video_files)}  video files. Starting processing...")


    for video_path in video_files:


        video_file_name = os.path.basename(video_path)

        video_name_no_ext = os.path.splitext(video_file_name)[0]



        output_folder = os.path.join(source_path, video_name_no_ext)



        os.makedirs(output_folder, exist_ok=True)

        print(f"\n--- Processing video: {video_file_name} ---")
        print(f"Frames will be saved to: {output_folder}")


        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: unable to open video file {video_file_name}")
            continue

        frame_idx = 0


        while True:



            ret, frame = cap.read()


            if not ret:
                break

            frame_idx += 1




            frame_name = f"{frame_idx:06d}.jpg"
            save_path = os.path.join(output_folder, frame_name)


            cv2.imwrite(save_path, frame)


            if frame_idx % 100 == 0:
                print(f"  Saved  {frame_idx}  frames...")


        cap.release()
        print(f"Processing completed: {video_file_name}. Saved  {frame_idx}  frames.")

    print("\n--- All videos have been processed ---")


if __name__ == "__main__":











    SOURCE_PATH = "/path/to/DynaMind-main/outputs/reconstruction/cine/failed"


    SAVE_PATH = SOURCE_PATH
    SAVE_PATH = "/path/to/cinebrain/dataset/test_clip"


    if SOURCE_PATH == "your_path/to/videos":
        print("Error: set the SOURCE_PATH variable in the script first.")
    else:
        extract_frames_from_folder(SOURCE_PATH)
