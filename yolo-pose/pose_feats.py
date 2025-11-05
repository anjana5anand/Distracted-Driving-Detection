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
from ultralytics import YOLO

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
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

dash_model.load_state_dict(torch.load('/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_32/dash/Resnext_dash_best.pth'))  # replace with actual path
# dash_model.load_state_dict(torch.load('weights/Resnext_dash_best.pth'))  # replace with actual path
dash_model.eval()
dash_model = dash_model.to(device)

rear_model = ResNetClassifier(num_classes=16)  # use correct num_classes
rear_model.load_state_dict(torch.load('/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_32/rear/Resnext_rear_best.pth'))  # replace with actual path
# dash_model.load_state_dict(torch.load('weights/Resnext_rear_best.pth'))  # replace with actual path
rear_model.eval()
rear_model = rear_model.to(device)

side_model = ResNetClassifier(num_classes=16)  # use correct num_classes
side_model.load_state_dict(torch.load('/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_32/side/Resnext_side_best.pth'))  # replace with actual path
# dash_model.load_state_dict(torch.load('weights/Resnext_side_best.pth'))  # replace with actual path
side_model.eval()
side_model = side_model.to(device)

pose_model = YOLO("yolo11x-pose.pt")


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


# def pose_feature_extractor(image, pose_model):
#     pose_features = pose_model(image)
#     for result in pose_features:
#         keypoints = pose_features.keypoints

#         kpts_array = keypoints.xy if hasattr(keypoints, 'xy') else keypoints.data.cpu().numpy()

#         keypoint_vectors = []
#         for instance_kpts in kpts_array:
#             print(instance_kpts)
#             flattened = instance_kpts.flatten()
#             keypoint_vectors.append(flattened)

#         if len(keypoint_vectors) > 0:
#             input_keypoints = keypoint_vectors[0]  # 1D vector for first detected person
#             print("Keypoint vector shape:", input_keypoints.shape)
#         else:
#             print("No keypoints detected in this frame.")
    

def pose_feature_extractor(gray, pose_model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_model.to(device)
    
    if len(gray.shape) == 2:
        rgb_img = torch.from_numpy(np.stack((gray,)*3, axis=-1))
    else:
        rgb_img = torch.from_numpy(gray)
    
    input_tensor = rgb_img.permute(2, 0, 1).float().to(device)
    input_tensor = input_tensor.unsqueeze(0) / 255.0 # add batch dimension
    
    with torch.no_grad():
        results = pose_model(input_tensor, verbose=False)
    
    # Find person with highest confidence
    if not results[0].boxes or len(results[0].boxes) == 0:
        return None
    
    # Get index of highest confidence detection
    confidences = results[0].boxes.conf
    highest_conf_idx = torch.argmax(confidences).item()
    
    final_keypoints = results[0].keypoints[highest_conf_idx].xy[0]
    final_keypoints = final_keypoints.flatten().unsqueeze(0)
    # print(final_keypoints.shape)
    # Return keypoints for most confident detection (still on GPU)
    return final_keypoints

def extract_features_from_video(video_path, model, pose_model, view_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every_n = int(fps // 5)

    frame_idx = 0
    pp = 0
    features = []
    pose_features = []
    print(fps, sample_every_n)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            pp += 1
            gray = crop_frame(frame, (512, 512), view_id)
    
            pose_feature = pose_feature_extractor(gray, pose_model)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # print(pose_feature.shape)
            # print(pose_feature)
            
            
            # pose_features.append(pose_feature)
            tensor = transform(gray).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.model(tensor)  # [1, feature_dim]
                if pose_feature is None or pose_feature.numel() == 0:
                    pose_feature = torch.zeros(1, 34).to(device)
                concat_feats = torch.cat((feat, pose_feature), axis = 1)
                
                # print(concat_feats.shape)
                # features.append(feat.squeeze(0).cpu().numpy())
                features.append(concat_feats.squeeze(0).cpu().numpy())

        frame_idx += 1
    print(video_path, frame_idx, pp)
    if not features:
        print(0, video_path)
    cap.release()
    
    # print(np.stack(features).shape)
    return np.stack(features) if features else np.empty((0, model.model.fc.in_features + 34))

video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/pose_resnet_features/A1"

os.makedirs(output_root, exist_ok=True)
for root, _, files in os.walk(video_root):
        # print(root, _, files)
        for file in files:
            if file.endswith('.MP4'):
                print(file)
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(video_path, video_root)
                save_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f'Processing {video_path}')
                if re.search('Dash', video_path):
                    features = extract_features_from_video(video_path, dash_model, pose_model, 0)
                elif re.search('Rear', video_path):
                    features = extract_features_from_video(video_path, rear_model, pose_model, 1)
                else:
                    features = extract_features_from_video(video_path, side_model, pose_model, 2)
                np.save(save_path, features)
                print(f'Saved features to {save_path}, shape={features.shape}')
