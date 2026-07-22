import os
from PIL import Image
import imageio.v3 as iio

def create_gif_from_images(input_dir: str, output_dir: str, images_per_gif: int = 6, duration_ms: int = 2000):
    """
    Combine every configured number of images from the input directory into a GIF and save the results in the output directory.

    Args:
        input_dir: Directory containing the source images.
        output_dir: Directory for generated GIF files.
        images_per_gif: Number of images in each GIF.
        duration_ms: Display duration of each frame in milliseconds.
    """
    print(f"--- Task started ---")


    if not os.path.isdir(input_dir):
        print(f"Error: input path does not exist: {input_dir}")
        return


    os.makedirs(output_dir, exist_ok=True)
    print(f"Output path is ready: {output_dir}")



    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])

    total_images = len(all_files)
    if total_images == 0:
        print("Warning: no image files were found in the input path.")
        return

    print(f"Found  {total_images}  images.")



    frame_duration_ms = duration_ms / images_per_gif
    print(f"Combine every  {images_per_gif}  images into one GIF with a total duration of  {duration_ms}ms。")
    print(f"The display duration of each image will be  {frame_duration_ms:.2f}  milliseconds.")


    gif_count = 0
    for i in range(0, total_images, images_per_gif):

        current_group_files = all_files[i:i + images_per_gif]


        if len(current_group_files) < 2:
            print(f"Notice: fewer than  {images_per_gif}  images remain; skipping.")
            break

        images_for_gif = []
        try:

            for filename in current_group_files:
                file_path = os.path.join(input_dir, filename)
                img = Image.open(file_path).convert('RGB')
                images_for_gif.append(img)


            gif_count += 1

            start_index = i + 1
            end_index = i + len(current_group_files)
            output_filename = f"group_{gif_count}_{start_index:04d}_to_{end_index:04d}.gif"
            output_path = os.path.join(output_dir, output_filename)


            iio.imwrite(
                output_path,
                images_for_gif,
                duration=frame_duration_ms,
                loop=0
            )

            print(f"Generated GIF successfully: {output_filename} ({len(images_for_gif)}  images)")

        except Exception as e:
            print(f"An error occurred while processing image group  {i} : {e}")
            continue

    print(f"--- Task completed; generated  {gif_count}  GIF files ---")



if __name__ == "__main__":

    input_path = "/path/to/choosed1/"
    output_path = "/path/to/exhibit_samples/"



    create_gif_from_images(
        input_dir=input_path,
        output_dir=output_path,
        images_per_gif=6,
        duration_ms=1500
    )
