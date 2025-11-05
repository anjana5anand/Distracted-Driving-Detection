from torch.utils.data import Dataset
import numpy as np
import os
import numpy as np
import csv
from datetime import timedelta
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

def frames_to_time(frame_idx, fps=5):
    """Convert frame index to HH:MM:SS timestamp string."""
    seconds = frame_idx / fps
    return str(timedelta(seconds=round(seconds)))

def smooth_predictions(preds, min_duration=3):
    """
    Merge small noisy segments in predictions with surrounding labels.

    Args:
        preds (np.ndarray): [T] per-frame predictions
        min_duration (int): minimum segment length to retain

    Returns:
        np.ndarray: smoothed predictions
    """
    preds = preds.copy()
    T = len(preds)
    start = 0

    while start < T:
        current_label = preds[start]
        end = start + 1
        while end < T and preds[end] == current_label:
            end += 1

        segment_length = end - start

        if segment_length < min_duration:
            left_label = preds[start - 1] if start > 0 else None
            right_label = preds[end] if end < T else None

            if left_label == right_label and left_label is not None:
                preds[start:end] = left_label

        start = end

    return preds

def predictions_to_csv(preds, output_csv_path, fps=5, ignore_class=0, min_duration=15):
    """
    Convert smoothed per-frame predictions to action segments and save as CSV.

    Args:
        preds (np.array): shape [num_frames], integer labels
        output_csv_path (str): path to save CSV
        fps (int): sampling rate (frames per second)
        ignore_class (int): background class label to ignore
        min_duration (int): minimum duration (in frames) to keep segments
    """
    # Smooth predictions
    preds = smooth_predictions(preds, min_duration=min_duration)

    num_frames = len(preds)
    segments = []
    start_idx = 0
    current_label = preds[0]

    for i in range(1, num_frames):
        if preds[i] != current_label:
            if current_label != ignore_class:
                segments.append((start_idx, i - 1, current_label))
            start_idx = i
            current_label = preds[i]

    if current_label != ignore_class:
        segments.append((start_idx, num_frames - 1, current_label))

    # Write to CSV
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Label"])
        for start, end, label in segments:
            start_time = frames_to_time(start, fps)
            end_time = frames_to_time(end, fps)
            writer.writerow([start_time, end_time, f"Class {label}"])

    print(f"Saved smoothed action segments to {output_csv_path}")

#2--------------------------------
def seconds_to_time_str(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

def predictions_to_csv(preds, output_csv_path, fps=5, ignore_class=0, min_duration=15):
    # Optional: median smoothing to remove single-frame glitches
    smoothed = preds.copy()
    for i in range(1, len(preds) - 1):
        if preds[i - 1] == preds[i + 1] != preds[i]:
            smoothed[i] = preds[i - 1]

    merged_segments = []
    start_idx = 0
    current_label = smoothed[0]

    for i in range(1, len(smoothed)):
        if smoothed[i] != current_label:
            duration = (i - start_idx) / fps
            if current_label != ignore_class and duration >= min_duration:
                start_time = seconds_to_time_str(start_idx / fps)
                end_time = seconds_to_time_str(i / fps)
                merged_segments.append([start_time, end_time, f"Class {current_label}"])
            start_idx = i
            current_label = smoothed[i]

    # Handle last segment
    duration = (len(smoothed) - start_idx) / fps
    if current_label != ignore_class and duration >= min_duration:
        start_time = seconds_to_time_str(start_idx / fps)
        end_time = seconds_to_time_str(len(smoothed) / fps)
        merged_segments.append([start_time, end_time, f"Class {current_label}"])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Label"])
        writer.writerows(merged_segments)
#3---------------
def seconds_to_time_str(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

def predictions_to_csv(preds, output_csv_path, fps=5, ignore_class=0, min_duration=15, max_gap=3):
    # Step 1: Optional smoothing to remove glitches
    smoothed = preds.copy()
    for i in range(1, len(preds) - 1):
        if preds[i - 1] == preds[i + 1] != preds[i]:
            smoothed[i] = preds[i - 1]

    # Step 2: Generate raw segments
    segments = []
    start_idx = 0
    current_label = smoothed[0]
    
    for i in range(1, len(smoothed)):
        if smoothed[i] != current_label:
            duration = (i - start_idx) / fps
            if current_label != ignore_class and duration >= min_duration:
                segments.append((start_idx, i, current_label))
            start_idx = i
            current_label = smoothed[i]

    # Handle last segment
    duration = (len(smoothed) - start_idx) / fps
    if current_label != ignore_class and duration >= min_duration:
        segments.append((start_idx, len(smoothed), current_label))

    # Step 3: Merge close segments of the same class
    merged_segments = []
    if segments:
        last_start, last_end, last_label = segments[0]
        for start, end, label in segments[1:]:
            gap = (start - last_end) / fps
            if label == last_label and gap <= max_gap:
                # Merge
                last_end = end
            else:
                merged_segments.append((last_start, last_end, last_label))
                last_start, last_end, last_label = start, end, label
        merged_segments.append((last_start, last_end, last_label))

    # Step 4: Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Label"])
        for start, end, label in merged_segments:
            writer.writerow([
                seconds_to_time_str(start / fps),
                seconds_to_time_str(end / fps),
                f"Class {label}"
            ])
#----------------
class MultiViewFeatureDataset(Dataset):
    def __init__(self, features_root, labels_root, views=("Dashboard", "Rear_view", "Right_side_window")):
        self.features_root = features_root
        self.labels_root = labels_root
        self.views = views
        self.sample_keys = []
        for user_folder in os.listdir(features_root):
            user_path = os.path.join(features_root, user_folder)
            if not os.path.isdir(user_path):
                continue
            for file in os.listdir(user_path):
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
        features_cat = np.concatenate((view_features[0][:min_rows, :], view_features[1][:min_rows, :], view_features[2][:min_rows, :]), axis=1)  # [seq_le>
        label_path = os.path.join(self.labels_root, os.path.dirname(key)+ ".npy")
        labels = np.load(label_path).astype(np.int64)  # [seq_len]
        file_name = key.split('/')[0]
        return file_name, features_cat, labels

class MambaSequenceClassifier(nn.Module):
    def __init__(self, input_dim=6144, hidden_dim=1024, num_classes=16, seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.mamba_block = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        returns: [batch_size, seq_len, num_classes]
        """
        x = self.input_proj(x)  # -> [B, L, H]
        x = self.mamba_block(x)  # [B, L, H]
        logits = self.output_layer(x)  # [B, L, C]
        return logits

def run_inference_and_generate_csv(model, dataset, output_dir, fps=5, min_duration=15, ignore_class=0, device="cuda"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for key, features, labels in loader:
        key = key[0]  # string
        features = features[0]  # [T, feat_dim]
        input_tensor = features.unsqueeze(0).to(device)  # [1, T, feat_dim]

        with torch.no_grad():
            logits = model(input_tensor)  # [1, T, num_classes]
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # [T]
        print(key)
        out_csv_path = os.path.join(output_dir, key.replace('/', '_') + ".csv")
        predictions_to_csv(preds, out_csv_path, fps=fps, ignore_class=ignore_class, min_duration=min_duration)

model = MambaSequenceClassifier(input_dim=6144, hidden_dim=256, num_classes=16)
# model.load_state_dict(torch.load("/home/viplab/Documents/mobilenetfinal/mamba_weights/mamba_best.pth"))
model.load_state_dict(torch.load("/media/viplab/DATADRIVE1/driver_action_recognition/MAMBA_TRAIN_WEIGHTS/mamba_weights_lr_5_chunk_100/mamba_best.pth"))
model.to("cuda")

# Inference dataset
test_dataset = MultiViewFeatureDataset(features_root="/media/viplab/Storage1/driver_action_recognition/raw_features/A1/validation", labels_root="/media/viplab/DATADRIVE1/driver_action_recognition/arrays")

# Run inference
run_inference_and_generate_csv(
    model=model,
    dataset=test_dataset,
    output_dir="/media/viplab/Storage1/driver_action_recognition/predict_csvs",
    fps=5,
    min_duration=3
)
