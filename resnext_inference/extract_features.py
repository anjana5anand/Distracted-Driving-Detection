import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision.datasets import ImageFolder
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import cv2
from datetime import timedelta
import pandas as pd
import random
import re

from torch.profiler import profile, record_function, ProfilerActivity

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # For grayscale
])

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

        first_conv_layer = self.model.conv1
        new_first_layer = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1 for grayscale input
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=False
        )

        with torch.no_grad():
            new_first_layer.weight[:] = first_conv_layer.weight.mean(dim=1, keepdim=True)

        self.model.conv1 = new_first_layer 
        num_features = self.model.fc.in_features
        self.classifier = nn.Linear(num_features, num_classes)
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dash_model = ResNetClassifier(num_classes=16)  # use correct num_classes

dash_model.load_state_dict(torch.load('/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_16/dash/Resnext_dash_best.pth'))  # replace with actual path
dash_model.eval()
dash_model = dash_model.to(device)

rear_model = ResNetClassifier(num_classes=16)  # use correct num_classes
rear_model.load_state_dict(torch.load('/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_16/rear/Resnext_rear_best.pth'))  # replace with actual path
rear_model.eval()
rear_model = rear_model.to(device)

side_model = ResNetClassifier(num_classes=16)  # use correct num_classes
side_model.load_state_dict(torch.load('/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_16/side/Resnext_side_best.pth'))  # replace with actual path
side_model.eval()
side_model = side_model.to(device)
# def crop_frame(frame, crop_size):
#         # height, width = frame.shape[:2]
        
#         # Calculate the starting x-coordinate for the right half
#         start_x = 500
#         end_x = 1500
        
#         # Crop the right half of the frame
#         cropped_frame = frame[:, start_x:, :]
        
#         # Resize the cropped frame to 512x512
#         resized_frame = cv2.resize(cropped_frame, crop_size)
        
#         return resized_frame

def crop_frame(frame, crop_size, view_filter):
    # height, width = frame.shape[:2]
    
    # Calculate the starting x-coordinate for the right half
    
    if(view_filter == 0):
        start_x = 200
        end_x = start_x + 1700
        
        start_y = 200
        end_y = start_y + 1700
    
    if(view_filter == 1):
        start_x = 600
        end_x = start_x + 1700
        
        start_y = 200
        end_y = start_y + 1700
        
    if(view_filter == 2):
        start_x = 700
        end_x = start_x + 1700
        
        start_y = 300
        end_y = start_y + 1700

    cropped_frame = frame[start_y:end_y, start_x:end_x]
    resized_frame = cv2.resize(cropped_frame, crop_size)
    
    return resized_frame

def extract_features_from_video(video_path, model, view_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every_n = int(fps // 5)

    frame_idx = 0
    pp = 0
    features = []
    print(fps, sample_every_n)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            pp += 1
            gray = crop_frame(frame, (224, 224), view_id)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(gray)
            tensor = transform(gray).unsqueeze(0).to(device)
            with torch.no_grad():
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
                    feat = model.model(tensor)  # [1, feature_dim]

                print(prof.key_averages().table(row_limit=10, sort_by="self_cpu_memory_usage"))
                os.system("sleep 5")
                # quit()
                features.append(feat.squeeze(0).cpu().numpy())

        frame_idx += 1
    print(video_path, frame_idx, pp)
    if not features:
        print(0, video_path)
    cap.release()
    return np.stack(features) if features else np.empty((0, model.model.fc.in_features))
print('22222')
video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
output_root = "/media/viplab/Storage1/driver_action_recognition/raw_features_test/A1"

os.makedirs(output_root, exist_ok=True)
l = ['20931']
for root, _, files in os.walk(video_root):
        # print(root, _, files)
        for file in files:
            if file.endswith('.MP4'):
                for i in l:
                    if i in file:
                        video_path = os.path.join(root, file)
                        rel_path = os.path.relpath(video_path, video_root)
                        save_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.npy')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        print(f'Processing {video_path}')
                        if re.search('Dash', video_path):
                            features = extract_features_from_video(video_path, dash_model, 0)
                        elif re.search('Rear', video_path):
                            features = extract_features_from_video(video_path, rear_model, 1)
                        else:
                            features = extract_features_from_video(video_path, side_model, 2)
                        # np.save(save_path, features)
                        print(f'Saved features to {save_path}, shape={features.shape}')


# for root, _, files in os.walk(video_root):
#         # print(root, _, files)
#         for file in files:
#             if file.endswith('.MP4'):
#                 print(file)
#                 # video_path = os.path.join(root, file)
#                 # rel_path = os.path.relpath(video_path, video_root)
#                 # save_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.npy')
#                 # os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                 # print(f'Processing {video_path}')
#                 # if re.search('Dash', video_path):
#                 #     features = extract_features_from_video(video_path, dash_model, 0)
#                 # elif re.search('Rear', video_path):
#                 #     features = extract_features_from_video(video_path, rear_model, 1)
#                 # else:
#                 #     features = extract_features_from_video(video_path, side_model, 2)
#                 # np.save(save_path, features)
#                 # print(f'Saved features to {save_path}, shape={features.shape}')
