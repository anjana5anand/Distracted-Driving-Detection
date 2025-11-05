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

from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        first_conv_layer = self.model.conv1
        new_first_layer = nn.Conv2d(1, first_conv_layer.out_channels,
                                    kernel_size=first_conv_layer.kernel_size,
                                    stride=first_conv_layer.stride,
                                    padding=first_conv_layer.padding, bias=False)
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


def crop_frame(frame, crop_size, view_filter):
    # height, width = frame.shape[:2]
    
    # Calculate the starting x-coordinate for the right half
    
    if(view_filter == "Dash"):
        start_x = 200
        end_x = start_x + 1700
        
        start_y = 200
        end_y = start_y + 1700
    
    if(view_filter == "Rear"):
        start_x = 600
        end_x = start_x + 1700
        
        start_y = 200
        end_y = start_y + 1700
        
    if(view_filter == "Side"):
        start_x = 700
        end_x = start_x + 1700
        
        start_y = 300
        end_y = start_y + 1700

    cropped_frame = frame[start_y:end_y, start_x:end_x]
    resized_frame = cv2.resize(cropped_frame, crop_size)
    
    return resized_frame



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
    
    final_keypoints = results[0].keypoints[highest_conf_idx].xyn[0]
    final_keypoints = final_keypoints.flatten().unsqueeze(0)
    # print(final_keypoints.shape)
    # Return keypoints for most confident detection (still on GPU)
    return final_keypoints



def extract_features_from_video(video_path, model_path, view):
    model = ResNetClassifier(num_classes=16)
    model.load_state_dict(torch.load(model_path))
    model.eval().to(device)

    pose_model = YOLO("yolo11x-pose.pt")

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
            gray = crop_frame(frame, (512, 512), view)
    
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
    # print(video_path, frame_idx, pp)
    if not features:
        print(0, video_path)
    cap.release()
    
    # print(np.stack(features).shape)
    return np.stack(features) if features else np.empty((0, model.fc.in_features + 34))



def process_video(args):
    video_path, video_root, output_root = args
    try:
        rel_path = os.path.relpath(video_path, video_root)
        save_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if re.search('Dash', video_path):
            model_path = '/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_16/dash/Resnext_dash_best.pth'
            # model_path = "/home/santhi/mohit/data_mount/final_resnet_weights/Model_dash_best.pth"
            view = 'Dash'
        elif re.search('Rear', video_path):
            model_path = '/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_16/rear/Resnext_rear_best.pth'
            # model_path = "/home/santhi/mohit/data_mount/final_resnet_weights/Model_rear_best.pth"
            view = 'Rear'
        else:
            # model_path = '/home/viplab/Documents/mobilenetfinal/resnet/side_weights/Model_side_best.pth'
            model_path = '/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights_batch_16/side/Resnext_side_best.pth'
            view = 'Side'



        print(f'[{view}] Processing {video_path}')
        features = extract_features_from_video(video_path, model_path, view)
        np.save(save_path, features)
        print(f'Saved to {save_path}, shape={features.shape}')
    except Exception as e:
        print(f'Failed {video_path}: {e}')

# if __name__ == '__main__':
#     video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
#     output_root = "/media/viplab/Storage1/driver_action_recognition/raw_features/A1_changed"
#     os.makedirs(output_root, exist_ok=True)

#     video_files = []
#     for root, _, files in os.walk(video_root):
#         for file in files:
#             if file.endswith('.MP4'):
#                 video_files.append(os.path.join(root, file))

#     print(f'Total videos: {len(video_files)}')

#     args = [(video_path, video_root, output_root) for video_path in video_files]

#     # Use number of logical cores or slightly less to avoid GPU overuse
#     with Pool(processes=min(cpu_count(), 4)) as pool:
#         list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # <-- Add this line


    video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/pose_resnet_features_norm/A1"

    print(os.listdir(video_root))
    os.makedirs(output_root, exist_ok=True)

    video_files = []
    existing_files = os.listdir('/media/viplab/DATADRIVE1/driver_action_recognition/pose_resnet_features_norm/A1')
    # existing_files.remove('user_id_38479_5')
    for root, _, files in os.walk(video_root):
        # print(root, _, files)
        for file in files:
            ch = "_".join(file.split('_')[-5:-2]) + '_' + file.split('_')[-1].split('.')[0]
            # print(ch)
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

    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 6)) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))
