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

from datetime import datetime
import pytz 

num_classes = 16
frame_size = (512, 512)
batch_size = 8
epochs = 100
learning_rate = 2e-6
weights_save_dir = "/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights/all"
view = "all"
all_images_paths = "/media/viplab/Storage1/driver_action_recognition/crop_new/cut_frames_dash"
log_file_path = "/media/viplab/DATADRIVE1/driver_action_recognition/resnext_weights/all_resnext/logs_resnet.txt"

ist = pytz.timezone('Asia/Kolkata')
#/santhi/driver_action_recgonition/data_mount/r

# image_paths = []
# labels = []
image_paths_train = []
labels_train = []
image_paths_test = []
labels_test = []
user_ids = set()


transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
 


class GreyscaleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # '/media/viplab/Storage1/driver_action_recognition/crop_new/cut_frames_dash/class_15/user_id_25432/Dashboard_user_id_25432_5_6919.jpg
        dimg_path = self.image_paths[idx]
        label = self.labels[idx]
        parts = dimg_path.split("/")
        rimg_path = "/".join(parts[:6]) + "/cut_frames_rear/" + "/".join(parts[7:-1]) + "/" + parts[-1].replace("Dashboard_user_id", "Rear_view_user_id")
        simg_path = "/".join(parts[:6]) + "/cut_frames_side/" + "/".join(parts[7:-1]) + "/" + parts[-1].replace("Dashboard_user_id", "Right_side_window_user_id")
        # if not os.path.exists(rimg_path) or not os.path.exists(simg_path):
        #     return None
        try:
            image_d = Image.open(dimg_path).convert('L')  # Convert to greyscale
            image_r = Image.open(rimg_path).convert('L')
            image_s = Image.open(simg_path).convert('L')
            if self.transform:
                image_d = self.transform(image_d)
                image_r = self.transform(image_r)
                image_s = self.transform(image_s)
            return image_d, image_r, image_s, label
        except Exception as e:
            return None
        # except:
        #     idx = np.random.randint(0, len(self.labels)-1)
            


for class_folder in os.listdir(all_images_paths):
    class_folder_path = os.path.join(all_images_paths, class_folder)
    if os.path.isdir(class_folder_path):
        for user_folder in os.listdir(class_folder_path):
            user_folder_path = os.path.join(class_folder_path, user_folder)
            if os.path.isdir(user_folder_path):
                user_ids.add(user_folder)
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

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



train_dataset = GreyscaleImageDataset(X_train, y_train, transform=transform)
val_dataset = GreyscaleImageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



model = ResNetClassifier(num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


data = []

print("Training Started")

with open(log_file_path, "a") as log_file:
        log_file.write(", ".join(['Epoch', 'TLoss', "TAccuracy", 'VLoss', 'VAccuracy']))
        log_file.write("\n")

for epoch in range(epochs):
    d = []
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for framesd, framesr, framess, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False):
        framesd, framesr, framess, labels = framesd.cuda(), framesr.cuda(), framess.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(framesd, framesr, framess)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / total:.4f}, Accuracy: {100. * correct / total:.2f}%")

    d.extend([epoch + 1, train_loss / total, 100 * correct / total])
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for framesd, framesr, framess, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation", leave=False):
            framesd, framesr, framess, labels = framesd.cuda(), framesr.cuda(), framess.cuda(), labels.cuda()
            outputs = model(framesd, framesr, framess)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss / total:.4f}, Accuracy: {val_accuracy:.2f}%")

    d.extend([val_loss / total, val_accuracy])
    data.append(d)
    
    # Save best weights validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), weights_save_dir + f"Resnext_{view}_best.pth")
        print(f"New best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.2f}%")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), weights_save_dir + f"Resnext_{view}_{epoch + 1}_{learning_rate}.pth")

    # Final Save
    torch.save(model.state_dict(), weights_save_dir + f"Resnext_final.pth")
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")
    os.system(f'echo "Best dash Weight {best_epoch}" > best_dash_epoch.txt')

    with open(log_file_path, "a") as log_file:
        current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        log_file.write(f"[{current_time}]")
        log_file.write(", ".join(map(str, d)))
        log_file.write("\n")