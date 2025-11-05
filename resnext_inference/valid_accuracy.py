from torchvision.datasets import ImageFolder
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm

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
    
num_classes = 16
frame_size = (512, 512)
validation_split = 0.2
all_images_paths = "/media/viplab/DATADRIVE1/driver_action_recognition/mobilenet_approach/cut_frames_dash"
image_paths_train = []
labels_train = []
image_paths_test = []
labels_test = []
user_ids = set()
batch_size = 32

transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for user_dir in os.listdir(class_path):
                    user_path = os.path.join(class_path, user_dir)
                    if os.path.isdir(user_path):
                        for img_file in os.listdir(user_path):
                            img_path = os.path.join(user_path, img_file)
                            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                                self.image_paths.append(img_path)
                                self.labels.append(class_idx)  # Class label based on folder name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # Convert to greyscale (L mode)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class GreyscaleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # Convert to greyscale

        if self.transform:
            image = self.transform(image)

        return image, label


for class_folder in os.listdir(all_images_paths):
    class_folder_path = os.path.join(all_images_paths, class_folder)
    if os.path.isdir(class_folder_path):
        for user_folder in os.listdir(class_folder_path):
            user_folder_path = os.path.join(class_folder_path, user_folder)
            if os.path.isdir(user_folder_path):
                user_ids.add(user_folder)


user_ids = sorted(user_ids)
#First 60
train_users = user_ids[:60]
#Last 9
test_users = user_ids[60:]   # Last 9 users for testing

for class_folder in os.listdir(all_images_paths):
    class_folder_path = os.path.join(all_images_paths, class_folder)
    if os.path.isdir(class_folder_path):
        label = int(class_folder.split('_')[1])  # Extract class number
        for user_folder in os.listdir(class_folder_path):
            user_folder_path = os.path.join(class_folder_path, user_folder)
            if os.path.isdir(user_folder_path):
                for img_file in os.listdir(user_folder_path):
                    if img_file.endswith(('png', 'jpg', 'jpeg')):
                        img_path = os.path.join(user_folder_path, img_file)
                        if user_folder in train_users:
                            image_paths_train.append(img_path)
                            labels_train.append(label)
                        elif user_folder in test_users:
                            image_paths_test.append(img_path)
                            labels_test.append(label)

X_train = image_paths_train
y_train = labels_train
X_test = image_paths_test
y_test = labels_test
model = ResNetClassifier(num_classes=num_classes).cuda()

train_dataset = GreyscaleImageDataset(X_train, y_train, transform=transform)
val_dataset = GreyscaleImageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    checkpoint = torch.load('/home/viplab/Documents/mobilenetfinal/resnet/resnet/Resnet_dash_20_1e-06.pth')
    model.load_state_dict(checkpoint)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Evaluation", leave =False):
            frames, labels = frames.cuda(), labels.cuda()
            outputs = model(frames)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # print(all_preds, all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    report = classification_report(all_labels, all_preds,
                                   target_names = [f"Class_{i}" for i in range(16)],
                                   digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm, all_preds, all_labels

p, pp, ppp, pppp = evaluate(model, val_loader, device='cuda')

print(p)
# print(pp)

#               precision    recall  f1-score   support                                                                                                     

#      Class_0     0.5141    0.2118    0.3000      1638
#      Class_1     0.6356    0.4738    0.5429       648
#      Class_2     0.9074    0.9778    0.9413      1173
#      Class_3     0.9486    0.8455    0.8941      1113
#      Class_4     0.3597    0.3592    0.3595      1428
#      Class_5     0.2729    0.3607    0.3108      1253
#      Class_6     0.7227    0.4464    0.5519      1483
#      Class_7     0.2587    0.6606    0.3718       383
#      Class_8     0.4572    0.2415    0.3161       973
#      Class_9     0.2156    0.4063    0.2817       443
#     Class_10     0.3591    0.3193    0.3380       758
#     Class_11     0.5000    0.4228    0.4582      1263
#     Class_12     0.4425    0.3126    0.3664      1353
#     Class_13     0.3450    0.6954    0.4612       453
#     Class_14     0.8995    0.9300    0.9145      1328
#     Class_15     0.2201    0.3696    0.2759      1423

#     accuracy                         0.4857     17113
#    macro avg     0.5037    0.5021    0.4803     17113
# weighted avg     0.5322    0.4857    0.4900     17113


#               precision    recall  f1-score   support                                                                                                     

#      Class_0     0.4750    0.2381    0.3172      1638
#      Class_1     0.7784    0.4552    0.5745       648
#      Class_2     0.9185    0.9804    0.9485      1173
#      Class_3     0.9549    0.8742    0.9128      1113
#      Class_4     0.3905    0.3557    0.3723      1428
#      Class_5     0.2410    0.4174    0.3056      1253
#      Class_6     0.6932    0.4983    0.5798      1483
#      Class_7     0.4553    0.5457    0.4964       383
#      Class_8     0.4252    0.3505    0.3842       973
#      Class_9     0.4631    0.3115    0.3725       443
#     Class_10     0.5550    0.2797    0.3719       758
#     Class_11     0.4249    0.4212    0.4231      1263
#     Class_12     0.4741    0.3999    0.4338      1353
#     Class_13     0.6817    0.6667    0.6741       453
#     Class_14     0.8999    0.9202    0.9099      1328
#     Class_15     0.2606    0.5439    0.3524      1423

#     accuracy                         0.5171     17113
#    macro avg     0.5682    0.5162    0.5268     17113
# weighted avg     0.5596    0.5171    0.5225     17113