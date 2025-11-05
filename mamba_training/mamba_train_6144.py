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


# skip_ids = [16700, 38159, 59359]
train_features_root = '/media/viplab/Storage1/driver_action_recognition/raw_features/A1/train'
train_labels_root = '/home/viplab/Documents/driver_action_recognition/data_processing/array_generation/arrays'

validation_features_root = '/media/viplab/Storage1/driver_action_recognition/raw_features/A1/valid'
validation_labels_root = '/home/viplab/Documents/driver_action_recognition/data_processing/array_generation/arrays'

num_epochs = 100
weights_save_path = "mamba_weights_6144_boundary"


class MultiViewFeatureDataset(Dataset):
    def __init__(self, features_root, labels_root, views=("Dashboard", "Rear_view", "Right_side_window")):
        self.features_root = features_root
        self.labels_root = labels_root
        self.views = views
        # self.skip_ids = ['16700', '38159', '59359']
        self.sample_keys = []
        print(features_root)
        for user_folder in os.listdir(features_root):
            # for i in self.skip_ids:
            #     if i in user_folder:
            #         continue
            print('uuuu', user_folder)
            user_path = os.path.join(features_root, user_folder)
            # print(user_path)
            if not os.path.isdir(user_path):
                continue
            for file in os.listdir(user_path):
                # print('f', file)
                if file.startswith("Dash") and file.endswith(".npy"):
                    print('1', os.path.splitext(file)[0])
                    key = os.path.join(user_folder, os.path.splitext(file)[0]) 
                    print('k', key) # e.g. user_001_1/dash_1
                    self.sample_keys.append(key)

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        key = self.sample_keys[idx]

        view_features = []
        for view in self.views:
            pp = view + '_' + "_".join(key.split('_')[:3]) + "_NoAudio_" + f"{key.split('_')[-1]}"
            path = os.path.join(self.features_root, os.path.dirname(key), f"{pp}.npy")
            features = np.load(path).astype(np.float32)  # [seq_len, feat_dim]
            view_features.append(features)
            # print('111111', features.shape, path)

        min_rows = min(view_features[0].shape[0], view_features[1].shape[0], view_features[2].shape[0])
        # Concatenate features across feature dimension
        features_cat = np.concatenate((view_features[0][:min_rows, :], view_features[1][:min_rows, :], view_features[2][:min_rows, :]), axis=1)  # [seq_len, total_feat_dim]
        label_path = os.path.join(self.labels_root, os.path.dirname(key)+ ".npy")
        labels = np.load(label_path).astype(np.int64)  # [seq_len]
        # print(features_cat.shape)
        return features_cat, labels

t_dataset = MultiViewFeatureDataset(
    features_root= train_features_root,
    labels_root= train_labels_root
)

v_dataset = MultiViewFeatureDataset(
    features_root= validation_features_root,
    labels_root= validation_labels_root
)
print(len(t_dataset))
print(len(v_dataset))

class ChunkedVideoDataset(Dataset):
    def __init__(self, base_dataset, chunk_size=100, stride=50):
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.stride = stride
        self.index_map = []  # (video_idx, start_frame)

        for video_idx in range(len(base_dataset)):
            features, labels = base_dataset[video_idx]
            video_len = features.shape[0]

            for start in range(0, video_len - chunk_size + 1, stride):
                labels_chunk = labels[start:start + chunk_size]

                if labels_chunk.sum() == 0:
                    continue

                self.index_map.append((video_idx, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        video_idx, start = self.index_map[idx]
        features, labels = self.base_dataset[video_idx]
        features_chunk = features[start:start+self.chunk_size]
        labels_chunk = labels[start:start+self.chunk_size]
        return features_chunk, labels_chunk

train_dataset = ChunkedVideoDataset(t_dataset, chunk_size=150, stride=75)
valid_dataset = ChunkedVideoDataset(v_dataset, chunk_size=150, stride=150)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=True
)

for i in train_dataset:
    print('train', len(i))
    break

for i in train_dataset:
    print('valid', len(i))
    break

for features, labels in train_loader:
    print('train')
    print("Concatenated features:", features.shape)  # [1, seq_len, total_feat_dim]
    print("Labels:", labels.shape)
    break 
for features, labels in valid_loader:
    print('valid')
    print("Concatenated features:", features.shape)  # [1, seq_len, total_feat_dim]
    print("Labels:", labels.shape)
    break 

class MambaSequenceClassifier(nn.Module):
    def __init__(self, input_dim=6144, hidden_dim=2048, num_classes=16, seq_len=100):
        super().__init__()
        # self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.mamba_block = Mamba(
            d_model=input_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.boundary_proj = nn.Sequential(                # For boundary regression
            nn.AdaptiveAvgPool1d(1),  # Reduces (B, H, T) -> (B, H, 1)
            nn.Flatten(start_dim=1),  # -> (B, H)
            nn.Linear(input_dim, 2),    # -> (B, 2): start and end
            nn.Sigmoid()              # Normalize between [0, 1]
        )
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        returns: [batch_size, seq_len, num_classes]
        """ 
        # x = self.input_proj(x)  # -> [B, L, H]
        x = self.mamba_block(x)  # [B, L, H]
        x_transposed = x.transpose(1, 2) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.output_layer(x)  # [B, L, C]
        boundaries = self.boundary_proj(x_transposed)
        return logits, boundaries
    
class BoundaryAwareLoss(nn.Module):
    def __init__(self, classification_weight=1.0, localization_weight=1.0, ignore_index=None):
        super().__init__()
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.ignore_index = ignore_index

    def forward(self, frame_logits, labels, pred_boundaries, true_boundaries):
        B, T, C = frame_logits.shape

        # Frame-wise classification loss
        loss_cls = F.cross_entropy(
            frame_logits.view(-1, C),
            labels.view(-1),
            ignore_index=self.ignore_index if self.ignore_index is not None else -100
        )

        # Temporal boundary regression loss (L1)
        loss_loc = F.l1_loss(pred_boundaries, true_boundaries)

        return self.classification_weight * loss_cls + self.localization_weight * loss_loc
def extract_boundaries_from_labels(labels, background_class=0):
    B, T = labels.shape
    boundaries = torch.zeros(B, 2)

    for b in range(B):
        action_mask = labels[b] != background_class
        indices = action_mask.nonzero(as_tuple=True)[0]

        if len(indices) > 0:
            start = indices[0].item()
            end = indices[-1].item()
            # Normalize
            boundaries[b, 0] = start / T
            boundaries[b, 1] = (end + 1) / T  # +1 to make end exclusive
        else:
            # If only background: set to [0, 0]
            boundaries[b, :] = 0.0

    return boundaries

def train_mamba(
    model, train_loader, val_loader,criterion, optimizer, num_epochs, weights_save_path, device,
    num_classes=16, ignore_index=None
):
    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_count = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = features.to(device)  # [B, 50, 6144]
            labels = labels.to(device)      # [B, 50]
            true_boundaries = extract_boundaries_from_labels(labels)  # [B, 2]
            true_boundaries = true_boundaries.to(device)
            optimizer.zero_grad()
            logits, pred_boundaries = model(features)        # [B, 50, num_classes]
            loss = criterion(logits, labels, pred_boundaries, true_boundaries)
            
            loss = F.cross_entropy(
                logits.view(-1, num_classes),
                labels.view(-1),
                # ignore_index=ignore_index if ignore_index is not None else -100,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- Accuracy ---
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
            for features, labels in tqdm(val_loader, desc="Validation"):
                features = features.to(device)
                labels = labels.to(device)

                logits, pred_boundaries = model(features)
                loss = criterion(logits, labels, pred_boundaries, pred_boundaries)
                
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_count += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_count if val_count > 0 else 0.0
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model checkpoint
            torch.save(model.state_dict(), f"{weights_save_path}/mamba_best.pth")
            print(f"Best model saved with accuracy: {val_accuracy*100:.2f}%")

        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy*100:.2f}%")
        # Save the model checkpoint
        if (epoch + 1) % 5 == 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), f"{weights_save_path}/mamba_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

model = MambaSequenceClassifier(input_dim=6144, hidden_dim=2048, num_classes=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
criterion = BoundaryAwareLoss(classification_weight=1.0, localization_weight=1.0)

train_mamba(model, train_loader, valid_loader, criterion, optimizer, num_epochs, weights_save_path, device="cuda", num_classes=16)

