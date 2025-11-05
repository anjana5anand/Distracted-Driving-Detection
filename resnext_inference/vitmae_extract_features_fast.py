#Hi
import os
import cv2
import torch
import numpy as np
import re
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
from transformers import ViTMAEForPreTraining, ViTImageProcessor
from transformers import AutoImageProcessor, ViTMAEModel
from PIL import Image
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cache = {}
pca = PCA(n_components = 72)

def crop_frame(frame, crop_size=(512, 512)):
    start_x = 500
    cropped_frame = frame[:, start_x:, :]
    resized_frame = cv2.resize(cropped_frame, crop_size)
    return resized_frame

def load_model(model_path):
    if model_path not in model_cache:
        # model = ViTMAEModel.from_pretrained(model_path).eval().to(device)
        model = ViTMAEForPreTraining.from_pretrained(model_path, mask_ratio=0, torch_dtype=torch.float16).eval().to(device)
        processor = AutoImageProcessor.from_pretrained(model_path)
        model_cache[model_path] = (model, processor)
    return model_cache[model_path]

model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/outputs/checkpoint-160000'
load_model(model_path)
model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/rear_outputs/checkpoint-130000'
load_model(model_path)
model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/side_outputs/checkpoint-120000'
load_model(model_path)

def extract_features_from_video(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every_n = max(int(fps // 5), 1)

    model, processor = load_model(model_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features = []

    for frame_idx in range(0, total_frames, sample_every_n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # print(f'[INFO] Processing frame {frame_idx}/{total_frames} from {video_path}')
        ret, frame = cap.read()
        if not ret:
            break

        cropped = crop_frame(frame)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=torch.float16) for k, v in inputs.items()} #, dtype=torch.float16

        with torch.no_grad():
            outputs = model(**inputs)
            # embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = outputs.logits
            embeddings = pca.fit_transform(embeddings.squeeze(0).cpu().numpy())
            features.append(embeddings)

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

        elif re.search('Rear', video_path):
            model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/rear_outputs/checkpoint-130000'

        else:
            model_path = '/home/viplab/Documents/driver_action_recognition/mae_training/transformers/examples/pytorch/image-pretraining/side_outputs/checkpoint-120000'

        print(f'[INFO] Processing [{model_path}] {video_path}')
        features = extract_features_from_video(video_path, model_path)
        np.save(save_path, features)
        print(f'[INFO] Saved: {save_path} | shape: {features.shape}')

    except Exception as e:
        print(f'[ERROR] Failed {video_path}: {e}')

def collect_video_files(video_root, output_root):
    video_files = []
    existing_set = set(f for f in os.listdir(output_root))
    print(len(existing_set))
    for root, _, files in os.walk(video_root):
        for file in files:
            if not file.endswith('.MP4'):
                continue
            file_path = os.path.join(root, file)
            pp = root.split('/')[-1]
            if pp in existing_set:
                continue
            ch = "_".join(file.split('_')[-5:-2]) + '_' + file.split('_')[-1].split('.')[0]
            video_files.append(file_path)

    return video_files



if __name__ == '__main__':
    set_start_method('spawn', force=True)  # <-- Add this line

    video_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_video/train/Dash"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_feat_vit_cls/train/Dash"
    os.makedirs(output_root, exist_ok=True)

    video_files = collect_video_files(video_root, output_root)
    print(f'Total videos: {len(video_files)}')

    args = [(vp, video_root, output_root) for vp in video_files]

    num_workers = min(cpu_count(), 4)  # You can tune this, depending on your GPU/CPU combo

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))

    video_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_video/train/Rear"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_feat_vit_cls/train/Rear"
    os.makedirs(output_root, exist_ok=True)

    video_files = collect_video_files(video_root, output_root)

    print(f'Total videos: {len(video_files)}')

    args = [(vp, video_root, output_root) for vp in video_files]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))

    video_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_video/train/Side"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_feat_vit_cls/train/Side"
    os.makedirs(output_root, exist_ok=True)

    video_files = collect_video_files(video_root, output_root)
    print(f'Total videos: {len(video_files)}')

    args = [(vp, video_root, output_root) for vp in video_files]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))

    set_start_method('spawn', force=True)  # <-- Add this line

    video_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_video/valid/Dash"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_feat_vit_cls/valid/Dash"
    os.makedirs(output_root, exist_ok=True)

    video_files = collect_video_files(video_root, output_root)
    print(f'Total videos: {len(video_files)}')

    args = [(vp, video_root, output_root) for vp in video_files]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))

    video_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_video/valid/Rear"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_feat_vit_cls/valid/Rear"
    os.makedirs(output_root, exist_ok=True)

    video_files = collect_video_files(video_root, output_root)
    print(f'Total videos: {len(video_files)}')

    args = [(vp, video_root, output_root) for vp in video_files]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))

    video_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_video/valid/Side"
    output_root = "/media/viplab/DATADRIVE1/driver_action_recognition/split_feat_vit_cls/valid/Side"
    os.makedirs(output_root, exist_ok=True)

    video_files = collect_video_files(video_root, output_root)
    print(f'Total videos: {len(video_files)}')

    args = [(vp, video_root, output_root) for vp in video_files]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))
