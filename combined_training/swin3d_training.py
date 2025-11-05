import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import cv2
import re
from mamba_ssm import Mamba
import torch.nn.functional as F
import torchvision.models.video as video_models

    
train_features_root = '/media/viplab/DATADRIVE1/driver_action_recognition/split_video/train/Dash'
validation_features_root = '/media/viplab/DATADRIVE1/driver_action_recognition/split_video/valid/Dash'

num_epochs = 100
weights_save_path = "mamba"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # For grayscale
])

class VideoDataset(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path
        self.samples = []
        for label in os.listdir(video_path):
            class_dir = os.path.join(video_path, label)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.endswith('.MP4'):
                        path = os.path.join(class_dir, f)
                        self.samples.append((path, int(label)))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        framed, framer, frames = self.load_video_frames(video_path)  # Shape: (seq_len, height, width, channels)
        
        return framed, framer, frames, label
    
    def load_video_frames(self, video_path):
        framed, framer, frames = [], [], []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = transform(gray)
            framed.append(tensor)
            frame_idx += 1
        cap.release()
        
        video_path = re.sub('Dashboard', 'Rear_view', video_path)
        video_path = re.sub('Dash', 'Rear', video_path)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = transform(gray)
            framer.append(tensor)
            frame_idx += 1
        cap.release()

        video_path = re.sub('Rear_view', 'Right_side_window', video_path)
        video_path = re.sub('Rear', 'Side', video_path)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = transform(gray)
            frames.append(tensor)
            frame_idx += 1
        if not frames or not framed or not framer:
            print(0, video_path)
        cap.release()
        # if np.stack(framed).shape[0] != 15 or np.stack(framer).shape[0] != 15 or np.stack(frames).shape[0] != 15:
        #     print(1, video_path)
        return np.stack(framed), np.stack(framer), np.stack(frames)

train_dataset = VideoDataset(
    video_path = train_features_root
)

valid_dataset = VideoDataset(
    video_path = validation_features_root
)
print(len(train_dataset))
print(len(valid_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=True
)

for featured, featurer, features, labels in train_loader:
    print('train')
    print("Concatenated features:", features)  # [1, seq_len, total_feat_dim]
    print("Labels:", labels)
    break 
for featured, featurer, features, labels in valid_loader:
    print('valid')
    print("Concatenated features:", features.shape)  # [1, seq_len, total_feat_dim]
    print("Labels:", labels.shape)
    break 

class Swin_3d_model(nn.Module):
    def __init__(self, hidden_dim = 24, num_classes = 16):
        super(Swin_3d_model, self).__init__()

        model = video_models.swin3d_t(weights="DEFAULT") 
        self.model = model
        self.model.head = torch.nn.Linear(model.head.in_features, num_classes)
        conv3d_weights = model.patch_embed.proj.weight  # Shape: (96, 3, 2, 4, 4)

        averaged_weights = conv3d_weights.mean(dim=1, keepdim=True)
        self.model.patch_embed.proj = nn.Conv3d(1, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4), bias=False)
        self.model.patch_embed.proj.weight.data = averaged_weights

        self.fc1 = nn.Linear(num_classes * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x, y, z):
        batch_size, seq_len, channel, height, width = x.size()
        
        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)
        z = z.permute(0, 2, 1, 3, 4)
        
        x = self.model(x)
        y = self.model(y)
        z = self.model(z)
        inp = torch.cat((x, y, z), 1)

        x = self.relu(self.fc1(inp))
        x = self.dropout(x)
        logits = self.output_layer(x)
        return logits
    
def train_swin3d(
    model, train_loader, val_loader, optimizer, num_epochs, weights_save_path, device='cuda',
    num_classes=16, ignore_index=None
):
    best_val_accuracy = 0.0
    model.train()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_count = 0

        for featured, featurer, features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            featured, featurer, features = featured.to(device), featurer.to(device), features.to(device)  # [B, 50, 6144]
            labels = labels.to(device)      # [B, 50]
            optimizer.zero_grad()
            logits = model(featured, featurer, features)        # [B, 50, num_classes]
            # print(logits.shape, labels.shape)
            loss = F.cross_entropy(
                logits.view(-1, num_classes),
                labels.view(-1),
                # ignore_index=ignore_index if ignore_index is not None else -100,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)  # [B, 50]
            # if ignore_index is not None:
            #     mask = labels != ignore_index
            #     total_correct += (preds[mask] == labels[mask]).sum().item()
            #     total_count += mask.sum().item()
            # else:
            total_correct += (preds == labels).sum().item()
            total_count += labels.numel()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_count if total_count > 0 else 0.0
        print(f"Epoch {epoch+1} -  Training Loss: {avg_train_loss:.4f} - Accuracy: {train_accuracy*100:.2f}%")
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for featured, featurer, features, labels in tqdm(val_loader, desc="Validation"):
                featured, featurer, features = featured.to(device), featurer.to(device), features.to(device)
                labels = labels.to(device)

                logits = model(featured, featurer, features)        # [B, 50, num_classes]
                # print('trainn', logits.shape, labels.shape)
                loss = F.cross_entropy(
                    logits.view(-1, num_classes),
                    labels.view(-1),
                    # ignore_index=ignore_index if ignore_index is not None else -100,
                )
                
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_count += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_count if val_count > 0 else 0.0
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model checkpoint
            # torch.save(model.state_dict(), f"{weights_save_path}/mamba_best.pth")
            print(f"Best model saved with accuracy: {val_accuracy*100:.2f}%")

        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy*100:.2f}%")
        # Save the model checkpoint
        if (epoch + 1) % 5 == 0:
            # Save the model checkpoint
            # torch.save(model.state_dict(), f"{weights_save_path}/mamba_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")
        # break

device = 'cuda'
model = Swin_3d_model(hidden_dim=24, num_classes=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

train_swin3d(model, train_loader, valid_loader, optimizer, num_epochs, weights_save_path, num_classes=16)

