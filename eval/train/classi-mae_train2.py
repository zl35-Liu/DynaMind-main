import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings


try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, AutoImageProcessor, VideoMAEConfig

    _has_transformers = True
    print("Hugging Face Transformers was found. VideoMAE from transformers will be used.")
except ImportError:
    _has_transformers = False
    print("Error: Hugging Face Transformers was not found. Install it with: pip install transformers imageio[ffmpeg]")
    exit()


torch.manual_seed(42)
np.random.seed(42)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


NUM_CLASSES = 40
BATCH_SIZE = 8
VAL_SPLIT = 0.1
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20




MODEL_STANDARD_IMAGE_SIZE = 224


NUM_FRAMES_VIDEOMAE = 6




VIDEO_MAE_PATCH_SIZE = (2, 16, 16)


LOCAL_MODEL_PATH = "./videoMAE/"

PRE_MODEL_PATH = "./checkpoints/best_40_class_videomae-1.pth"


warnings.filterwarnings("ignore",
                        message="The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.")
warnings.filterwarnings("ignore",
                        message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.")



class VideoMAEClassificationDataset(Dataset):
    def __init__(self, video_data, labels, model_path=LOCAL_MODEL_PATH,
                 num_frames=NUM_FRAMES_VIDEOMAE, model_input_image_size=MODEL_STANDARD_IMAGE_SIZE):
        """
        Dataset used to train VideoMAE. Video data has shape (samples, source frames, channels, height, width), labels have shape (samples,), and num_frames controls the sampled frame count.
        """
        self.video_data = video_data
        self.labels = labels
        self.num_frames = num_frames
        self.model_input_image_size = model_input_image_size

        assert len(self.video_data) == len(self.labels), "The video and label counts do not match."




        processor_config_obj = VideoMAEImageProcessor.from_pretrained(model_path)


        processor_config_obj.do_resize = True



        processor_config_obj.size = {"shortest_edge": model_input_image_size}
        processor_config_obj.do_center_crop = True
        processor_config_obj.crop_size = {"height": model_input_image_size, "width": model_input_image_size}
        processor_config_obj.num_frames = num_frames


        print(f"Loading and configuring ImageProcessor from local path: {model_path}")
        self.image_processor = VideoMAEImageProcessor(**processor_config_obj.to_dict())


    def __len__(self):
        return len(self.labels)

    def sample_frames(self, video_frames_np):
        """
        Uniformly sample the requested number of frames from a video. When the source already has the requested frame count, all frames are returned.
        """
        total_frames = video_frames_np.shape[0]
        if total_frames == 0:

            return np.zeros(
                (self.num_frames, self.video_data.shape[-2], self.video_data.shape[-1], self.video_data.shape[-3]),
                dtype=np.uint8)

        if total_frames < self.num_frames:
            indices = np.random.choice(total_frames, self.num_frames, replace=True)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        return video_frames_np[indices]

    def __getitem__(self, idx):


        video_clip = self.video_data[idx].transpose(0, 2, 3, 1)


        sampled_frames = self.sample_frames(video_clip)


        label = self.labels[idx]



        inputs = self.image_processor(list(sampled_frames), return_tensors="pt").pixel_values.squeeze(0)







        return inputs, torch.tensor(label, dtype=torch.long)



def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device).long()










        outputs = model(pixel_values=inputs).logits

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc



def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(pixel_values=inputs).logits

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = correct / total

    report = classification_report(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, report



def train_videomae_classifier(epochs=NUM_EPOCHS, lr=LEARNING_RATE, num_classes=NUM_CLASSES, num=None,
                              video_data=None, labels=None,
                              local_model_path=LOCAL_MODEL_PATH, pre_model_path=PRE_MODEL_PATH):
    """
    Train the VideoMAE classifier. The inputs configure epochs, learning rate, class count, video arrays, labels, batch size, output path, and the optional pretrained model path.
    """
    if not _has_transformers:
        print("Error: Hugging Face Transformers was not found, so the VideoMAE model cannot be trained.")
        return None


    dataset = VideoMAEClassificationDataset(
        video_data=video_data,
        labels=labels,
        num_frames=NUM_FRAMES_VIDEOMAE,
        model_input_image_size=MODEL_STANDARD_IMAGE_SIZE,
        model_path=local_model_path
    )

    dataset_size = len(dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count() // 2)

    print(f"Dataset size: total  {dataset_size}, training  {train_size}, validation  {val_size}")
    print(f"Training {num_classes}  classifier with VideoMAE")



    config = VideoMAEConfig.from_pretrained(local_model_path)


    config.num_frames = NUM_FRAMES_VIDEOMAE
    config.image_size = MODEL_STANDARD_IMAGE_SIZE
    config.patch_size = (VIDEO_MAE_PATCH_SIZE[1], VIDEO_MAE_PATCH_SIZE[2])
    config.tube_patch_size = VIDEO_MAE_PATCH_SIZE
    config.num_labels = num_classes

    print(
        f"Updated VideoMAE configuration: num_frames={config.num_frames}, image_size={config.image_size}, patch_size={config.patch_size}, tube_patch_size={config.tube_patch_size}, num_labels={config.num_labels}")


    model = VideoMAEForVideoClassification(config)



    if os.path.exists(pre_model_path):
        print(f"Loading from  {pre_model_path} Loading fine-tuned model weights...")

        model.load_state_dict(torch.load(pre_model_path, map_location='cpu'))
    else:
        print(f"Warning: fine-tuned weights were not found at  {pre_model_path}. Training will start from official pretrained weights when available, otherwise from random initialization.")


        try:
            temp_model = VideoMAEForVideoClassification.from_pretrained(
                local_model_path,
                config=config,
                ignore_mismatched_sizes=True
            )
            model.load_state_dict(temp_model.state_dict(), strict=False)
            print("Partial official pretrained weights were loaded.")
        except Exception as e:
            print(f"Failed to load official pretrained weights: {e}. The model will use randomly initialized weights.")

    model = model.to(device)


    print(f"Using VideoMAE; number of output classes: {num_classes}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    model_name = "40_class_videomae"

    best_model_path = f"./checkpoints/best_{model_name}-{num}.pth"

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print('-' * 20)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, report = validate(model, val_loader, criterion)

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Training loss: {train_loss:.4f} | accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved a new best model; validation accuracy: {val_acc:.4f}")

        if (epoch + 1) % 5 == 0:
            print(f"\n{num_classes}Detailed classification report:")
            print(report)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_metrics_{model_name}-{num}.png')
    plt.show()

    print("\nEvaluating the best model on the full dataset...")
    model.load_state_dict(torch.load(best_model_path))
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    _, full_acc, full_report = validate(model, full_loader, criterion)

    print(f"Final accuracy: {full_acc:.4f}")
    print(f"\n{num_classes}Detailed classification report:")
    print(full_report)

    result_file = f"{model_name}-{num}_results.txt"
    with open(result_file, "w") as f:
        f.write(f"{num_classes}Classification accuracy: {full_acc:.4f}\n\n")
        f.write("Detailed report:\n")
        f.write(full_report)

    print(f"Evaluation completed. Results saved to  {result_file}")

    return model



if __name__ == "__main__":

    video_data = np.load("/path/to/workspace/Ljx/EEG2Video-main/acessment/video_frames.npy")
    video_data = video_data[200:]




    print(f"Original video data shape: {video_data.shape}")
    print(f"Original video pixel-value range: {video_data.min()} - {video_data.max()}")


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
    for block_id in range(7):
        All_label = np.concatenate((All_label, GT_label[block_id].repeat(5).reshape(1, 200)))
    class_labels = rearrange(All_label, 'b c -> (b c)') - 1
    class_labels = class_labels[200:]
    print("40-class label shape:", class_labels.shape)
    print(f"Minimum label value: {class_labels.min()}")
    print(f"Maximum label value: {class_labels.max()}")
    print(
        f"Whether any label is outside [0, {NUM_CLASSES - 1}]: {(class_labels < 0).any() or (class_labels >= NUM_CLASSES).any()}")


    print("\nStarting training for the 40-class VideoMAE model...")

    model_40_videomae = train_videomae_classifier(
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        num_classes=NUM_CLASSES,
        num=1,
        video_data=video_data,
        labels=class_labels,
        local_model_path=LOCAL_MODEL_PATH,
        pre_model_path=PRE_MODEL_PATH
    )

    print("VideoMAE model training is complete.")
