"""
Train a ResNet-50 classifier using the middle frame of each video.
"""
import os
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import cv2



torch.manual_seed(42)
np.random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


NUM_CLASSES_40 = 40
BATCH_SIZE = 32
VAL_SPLIT = 0.1






class VideoFrameDataset(Dataset):
    def __init__(self, video_data=None, labels_40=None, labels_2=None, gif_dir=None, target_size=(288, 512)):
        """
        Create a dataset from either an in-memory video array and labels or a directory of GIF files and labels. Video arrays have shape (samples, frames, channels, height, width).
        """
        self.target_size = target_size
        self.labels_40 = labels_40
        self.labels_2 = labels_2


        if video_data is not None:
            self.video_data = video_data
            self.mode = "numpy"
            print(f"Using NumPy array input; video count: {len(video_data)}")
        elif gif_dir is not None:
            self.mode = "gif"
            self.gif_dir = gif_dir
            self.gif_paths = self.get_gif_paths()
            print(f"Using GIF input; GIF count: {len(self.gif_paths)}")
        else:
            raise ValueError("Either video_data or gif_dir must be provided")


        num_videos = len(self.video_data) if self.mode == "numpy" else len(self.gif_paths)
        if labels_40 is not None:
            assert len(labels_40) == num_videos, f"40-class label count ({len(labels_40)}) and video count ({num_videos}) do not match"
        if labels_2 is not None:
            assert len(labels_2) == num_videos, f"Binary label count ({len(labels_2)}) and video count ({num_videos}) do not match"

    def get_gif_paths(self):
        """
        Return all GIF file paths in the configured directory.
        """
        gif_files = [f for f in os.listdir(self.gif_dir) if f.endswith(".gif")]
        return [os.path.join(self.gif_dir, f) for f in gif_files]

    def __len__(self):
        if self.labels_40 is not None:
            return len(self.labels_40)
        elif self.labels_2 is not None:
            return len(self.labels_2)
        else:
            raise ValueError("At least one label type must be provided")

    def preprocess_frame(self, frame):
        """
        Preprocess a single image frame.
        """

        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)


        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


        frame = cv2.resize(frame, self.target_size) if frame.shape[:2] != self.target_size else frame


        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0


        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std


        frame = frame.transpose(2, 0, 1)
        return frame

    def preprocess_gif(self, gif_path):
        """
        Preprocess a GIF file and extract its middle frame.
        """

        gif = imageio.mimread(gif_path)


        mid_idx = len(gif) // 2
        frame = gif[mid_idx]


        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


        return self.preprocess_frame(frame)

    def preprocess_video(self, video):
        """
        Preprocess one video sample and extract its middle frame.
        """


        mid_idx = video.shape[0] // 2
        frame = video[mid_idx]
        return self.preprocess_frame(frame)

    def __getitem__(self, idx):

        if self.mode == "numpy":
            frame = self.preprocess_video(self.video_data[idx])
        else:
            frame = self.preprocess_gif(self.gif_paths[idx])


        frame_tensor = torch.tensor(frame).float()


        return_data = (frame_tensor,)

        if self.labels_40 is not None:
            return_data += (self.labels_40[idx],)

        if self.labels_2 is not None:
            return_data += (self.labels_2[idx],)

        return return_data



def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in tqdm(loader, desc="Training"):
        if len(data) == 2:
            inputs, labels = data
        else:
            raise ValueError("Invalid data format; expected data and labels")

        inputs = inputs.to(device)

        labels = labels.to(device).long()


        outputs = model(inputs)


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
        for data in tqdm(loader, desc="Validation"):
            if len(data) == 2:
                inputs, labels = data
            else:
                raise ValueError("Invalid data format; expected data and labels")

            inputs = inputs.to(device)

            labels = labels.to(device).long()


            outputs = model(inputs)


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



def train_resnet_classifier(epochs=30, lr=0.001, num_classes=40,train_idx=0,
                            video_data=None, labels=None, labels_2=None,
                            gif_dir=None):
    """
    Train a ResNet-50 classifier end to end. num_classes selects either the 40-class or binary task, and the corresponding label array must be provided.
    """

    if num_classes == NUM_CLASSES_40:
        dataset = VideoFrameDataset(video_data=video_data, labels_40=labels,
                                    labels_2=None, gif_dir=gif_dir)
    else:
        dataset = VideoFrameDataset(video_data=video_data, labels_40=None,
                                    labels_2=labels_2, gif_dir=gif_dir)


    dataset_size = len(dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Dataset size: total  {dataset_size}, training  {train_size}, validation  {val_size}")
    print(f"Training {num_classes}  classifier")


    model = torchvision.models.resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.num_classes = num_classes
    model = model.to(device)

    print(f"Using ResNet-50; number of output classes: {num_classes}")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)


    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    model_name = "40_class" if num_classes == 40 else "2_class"
    best_model_path = f"/path/to/DynaMind-main/outputs/eval/checkpoints/best_resnet_{model_name}-{train_idx}.pth"


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
    plt.savefig(f'training_metrics_{model_name}.png')
    plt.show()


    print("\nEvaluating the best model on the full dataset...")
    model.load_state_dict(torch.load(best_model_path))
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    _, full_acc, full_report = validate(model, full_loader, criterion)

    print(f"Final accuracy: {full_acc:.4f}")
    print(f"\n{num_classes}Detailed classification report:")
    print(full_report)


    result_file = f"resnet_{model_name}_results.txt"
    with open(result_file, "w") as f:
        f.write(f"{num_classes}Classification accuracy: {full_acc:.4f}\n\n")
        f.write("Detailed report:\n")
        f.write(full_report)

    print(f"Evaluation completed. Results saved to  {result_file}")

    return model



if __name__ == "__main__":

    video_data = np.load("/path/to/DynaMind-main/data/Video/video_tensor/all_video.npy")

    print(video_data.shape)
    video_data = video_data/255
    print(video_data.min())
    print(video_data.max())


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
    class_labels = class_labels
    print("40-class label shape:", class_labels.shape)








    print("\nTraining the 40-class model...")
    model_40 = train_resnet_classifier(
        epochs=20,
        lr=0.0001,
        num_classes=40,
        train_idx=2,
        video_data=video_data,
        labels=class_labels
    )













    print("All model training is complete.")
