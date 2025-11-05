import os
import cv2
import torch
import numpy as np
import re
from torchvision import transforms, models
from torch import nn
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import multiprocessing
from transformers import AutoFeatureExtractor, AutoModel, ViTMAEForPreTraining, ViTImageProcessor
from PIL import Image
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

model_name_or_path = "D:/SAM/Sem8/Capstone_Final_Project/transformers/examples/pytorch/image-pretraining/outputs/checkpoint-500"  # path to your checkpoint
model = ViTMAEForPreTraining.from_pretrained(model_name_or_path, mask_ratio=0)
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

model.eval()  

image_path = "D:/SAM/Sem8/Capstone_Final_Project/dash_frames_single_user/train/Dashboard_user_id_14786_5_385.jpg"
image = Image.open(image_path).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.logits

def crop_frame(frame, crop_size=(512, 512)):
    start_x = 500
    cropped_frame = frame[:, start_x:, :]
    resized_frame = cv2.resize(cropped_frame, crop_size)
    return resized_frame

def extract_features_from_video(video_path, model_path, view):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every_n = int(fps // 5)
    features = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            cropped = crop_frame(frame)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(gray)
            inputs = feature_extractor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.logits
                features.append(embeddings.squeeze(0).cpu().numpy())
        frame_idx += 1
    cap.release()
    return np.stack(features) if features else np.empty((0, model.model.fc.in_features))

def process_video(args):
    video_path, video_root, output_root = args
    try:
        rel_path = os.path.relpath(video_path, video_root)
        save_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if re.search('Dash', video_path):
            model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/outputs/checkpoint-160000'
            view = 'Dash'
        elif re.search('Rear', video_path):
            model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/rear_outputs/checkpoint-130000'
            view = 'Rear'
        else:
            model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/side_outputs/checkpoint-120000'
            view = 'Side'

        print(f'[{view}] Processing {video_path}')
        features = extract_features_from_video(video_path, model_path, view)
        np.save(save_path, features)
        print(f'Saved to {save_path}, shape={features.shape}')
    except Exception as e:
        print(f'Failed {video_path}: {e}')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # <-- Add this line

    video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
    output_root = "/media/viplab/Storage1/driver_action_recognition/raw_features/A1_changed"
    os.makedirs(output_root, exist_ok=True)

    video_files = []
    existing_files = os.listdir(output_root)
    for root, _, files in os.walk(video_root):
        for file in files:
            ch = "_".join(file.split('_')[-5:-2]) + '_' + file.split('_')[-1].split('.')[0]
            if ch in existing_files:
                continue
            if file.endswith('.MP4'):
                video_files.append(os.path.join(root, file))
    # for root, _, files in os.walk(video_root):
    #     for file in files:
    #         existing_files = os.listdir(output_root)
            
    #         if file.endswith('.MP4'):
    #             video_files.append(os.path.join(root, file))

    print(f'Total videos: {len(video_files)}')

    args = [(video_path, video_root, output_root) for video_path in video_files]

    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4)) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))
