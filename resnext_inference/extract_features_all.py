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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # For grayscale
])


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.modeld = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.modelr = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.models = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

        first_conv_layer = self.modeld.conv1
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

        self.modeld.conv1 = new_first_layer 
        self.modelr.conv1 = new_first_layer 
        self.models.conv1 = new_first_layer 

        self.feature_dim = self.modeld.fc.in_features * 3

        self.modeld.fc = nn.Identity()
        self.modelr.fc = nn.Identity()
        self.models.fc = nn.Identity()

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x1, x2, x3):
        x1 = self.modeld(x1)
        x2 = self.modelr(x2)
        x3 = self.models(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.classifier(x)
        return x
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetClassifier(num_classes=16)  # use correct num_classes
model.load_state_dict(torch.load('/media/viplab/Storage1/driver_action_recognition/crop_new/combined_weights/Resnext_all_best.pth'))  # replace with actual path
model.eval()
model = model.to(device)

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

def extract_features_from_video(video_path, id):
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
            gray = crop_frame(frame, (512, 512), id)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = transform(gray).unsqueeze(0).to(device)
            with torch.no_grad():
                if id == 0:
                    feat = model.modeld(tensor)  # [1, feature_dim]
                    features.append(feat.squeeze(0).cpu().numpy())
                elif id == 1:
                    feat = model.modelr(tensor)  # [1, feature_dim]
                    features.append(feat.squeeze(0).cpu().numpy())
                else:
                    feat = model.models(tensor)  # [1, feature_dim]
                    features.append(feat.squeeze(0).cpu().numpy())
        frame_idx += 1
    print(video_path, frame_idx, pp)
    if not features:
        print(0, video_path)
    cap.release()
    return np.stack(features) if features else np.empty((0, model.model.fc.in_features))
print('22222')
video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
output_root = "/media/viplab/Storage1/driver_action_recognition/raw_features_all/A1"
os.makedirs(output_root, exist_ok=True)
for root, _, files in os.walk(video_root):
        # print(root, _, files)
        for file in files:
            if file.endswith('.MP4'):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(video_path, video_root)
                save_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f'Processing {video_path}')
                if re.search('Dash', video_path):
                    features = extract_features_from_video(video_path, 0)
                elif re.search('Rear', video_path):
                    features = extract_features_from_video(video_path, 1)
                else:
                    features = extract_features_from_video(video_path, 2)
                
                np.save(save_path, features)
                print(f'Saved features to {save_path}, shape={features.shape}')

 