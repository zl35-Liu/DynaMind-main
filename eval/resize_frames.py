import os
from PIL import Image

def resize_images_direct(input_dir, output_dir, target_size=256):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")


        img = img.resize((target_size, target_size), Image.BICUBIC)


        out_path = os.path.join(output_dir, filename)
        img.save(out_path)

    print(f"All images resized (direct stretch) to {target_size}x{target_size} and saved in {output_dir}")

if __name__ == "__main__":
    resize_images_direct(
        input_dir="./outputs/DGVR/video/Clip1",
        output_dir="./outputs/DGVR/video/Clip1",
        target_size=256
    )
