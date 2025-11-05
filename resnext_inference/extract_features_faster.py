#Hi
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

def crop_frame(frame, crop_size=(512, 512)):
    start_x = 500
    cropped_frame = frame[:, start_x:, :]
    resized_frame = cv2.resize(cropped_frame, crop_size)
    return resized_frame

def extract_features_from_video(video_path, model_path, view):
    model = ResNetClassifier(num_classes=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

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
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            tensor = transform(gray).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.model(tensor)
                features.append(feat.squeeze(0).cpu().numpy())
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
            # model_path = '/home/viplab/Documents/mobilenetfinal/resnet/dash_weights/Resnet_dash_best.pth'
            model_path = "/home/santhi/mohit/data_mount/final_resnet_weights/Model_dash_best.pth"
            view = 'Dash'
        elif re.search('Rear', video_path):
            # model_path = '/home/viplab/Documents/mobilenetfinal/resnet/rear_weights/Model_rear_best.pth'
            model_path = "/home/santhi/mohit/data_mount/final_resnet_weights/Model_rear_best.pth"
            view = 'Rear'
        else:
            # model_path = '/home/viplab/Documents/mobilenetfinal/resnet/side_weights/Model_side_best.pth'
            model_path = "/home/santhi/mohit/data_mount/final_resnet_weights/Model_side_best.pth"
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

    # video_root = "/media/viplab/Storage1/driver_action_recognition/raw_videos/A1_changed"
    video_root = "/home/santhi/mohit/data_mount/raw_videos/A1_changed"
    # output_root = "/media/viplab/Storage1/driver_action_recognition/raw_features/A1_changed"
    output_root = "/home/santhi/mohit/data/extracted_features/feat_new"
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
