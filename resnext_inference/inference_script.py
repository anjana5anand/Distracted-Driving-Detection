import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from model import MambaModel  # your trained Mamba model
from resnet_classifier import ResNetClassifier  # your custom ResNet
from datetime import timedelta

# --- Config ---
FPS = 5
SEQUENCE_LENGTH = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Video to frames ---
def read_video_frames(video_path, fps=5):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(orig_fps // fps)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        count += 1
    cap.release()
    return frames

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Feature extraction ---
def extract_features(frames, model):
    model.eval()
    features = []
    with torch.no_grad():
        for frame in frames:
            x = preprocess(frame).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]
            feat = model.model(x)  # No classifier
            features.append(feat.squeeze(0).cpu().numpy())
    return np.stack(features)  # [T, D]

# --- Post-processing predictions ---
def smooth_and_merge(preds, threshold=2):
    preds = preds.tolist()
    final = []
    prev = preds[0]
    start = 0
    for i in range(1, len(preds)):
        if preds[i] != prev:
            if preds[i] != 0 and (i - start) <= threshold:
                # Fill short burst with previous class
                preds[start:i] = [prev] * (i - start)
            else:
                final.append((start, i - 1, prev))
                start = i
                prev = preds[i]
    final.append((start, len(preds) - 1, prev))

    # Merge consecutive segments of same class with short gaps
    merged = []
    for seg in final:
        if seg[2] == 0:
            continue
        if not merged:
            merged.append(seg)
        else:
            prev_seg = merged[-1]
            if seg[2] == prev_seg[2] and seg[0] - prev_seg[1] <= threshold:
                merged[-1] = (prev_seg[0], seg[1], seg[2])
            else:
                merged.append(seg)
    return merged

# --- Convert to HH:MM:SS ---
def to_time(seconds):
    return str(timedelta(seconds=seconds))

# --- Inference pipeline ---
def run_inference(dash_path, rear_path, side_path, 
                  dash_model, rear_model, side_model, mamba_model, output_csv):
    dash_frames = read_video_frames(dash_path, FPS)
    rear_frames = read_video_frames(rear_path, FPS)
    side_frames = read_video_frames(side_path, FPS)

    length = min(len(dash_frames), len(rear_frames), len(side_frames))
    dash_feat = extract_features(dash_frames[:length], dash_model)
    rear_feat = extract_features(rear_frames[:length], rear_model)
    side_feat = extract_features(side_frames[:length], side_model)

    full_feat = np.concatenate([dash_feat, rear_feat, side_feat], axis=1)  # [T, D]

    # Run model
    mamba_model.eval()
    with torch.no_grad():
        x = torch.tensor(full_feat).unsqueeze(0).to(DEVICE)  # [1, T, D]
        logits = mamba_model(x)  # [1, T, num_classes]
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    segments = smooth_and_merge(preds)

    # Write to CSV
    with open(output_csv, 'w') as f:
        for start, end, label in segments:
            start_time = to_time(start * (1 / FPS))
            end_time = to_time((end + 1) * (1 / FPS))
            f.write(f"{start_time},{end_time},Class {label}\n")
    print(f"Saved: {output_csv}")

# --- Load models ---
dash_model = ResNetClassifier(num_classes=15).to(DEVICE)
rear_model = ResNetClassifier(num_classes=15).to(DEVICE)
side_model = ResNetClassifier(num_classes=15).to(DEVICE)

# Load their weights
dash_model.load_state_dict(torch.load('dash_model.pth'))
rear_model.load_state_dict(torch.load('rear_model.pth'))
side_model.load_state_dict(torch.load('side_model.pth'))

mamba_model = MambaModel(input_dim=6144, num_classes=15).to(DEVICE)
mamba_model.load_state_dict(torch.load('mamba_model.pth'))

# Example usage
run_inference(
    dash_path="test_videos/dash.mp4",
    rear_path="test_videos/rear.mp4",
    side_path="test_videos/side.mp4",
    dash_model=dash_model,
    rear_model=rear_model,
    side_model=side_model,
    mamba_model=mamba_model,
    output_csv="inference_output.csv"
) 
