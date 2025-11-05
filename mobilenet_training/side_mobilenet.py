# %%
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

# %%
class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v3_large(weights=True)

        first_conv_layer = self.model.features[0][0]
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

        self.model.features[0][0] = new_first_layer 
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# %%
num_classes = 16
# frame_size = (224, 224)
frame_size = (512, 512)
batch_size = 64
epochs = 100
validation_split = 0.2
learning_rate = 1e-4
weight_decay_rate = 1e-4
weights_save_dir = "side_weights/"
best_val_accuracy = 0
view = "side"

# %%
transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

# %%
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

# %%
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

# %%
all_images_paths = "/media/viplab/DATADRIVE1/driver_action_recognition/mobilenet_approach/cut_frames_side"
image_paths = []
labels = []

image_paths_train = []
labels_train = []

image_paths_test = []
labels_test = []


user_ids = set()
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




# %%
for class_folder in os.listdir(all_images_paths):
    class_folder_path = os.path.join(all_images_paths, class_folder)
    if os.path.isdir(class_folder_path):
        label = int(class_folder.split('_')[1])  # Extract class number
        for user_folder in os.listdir(class_folder_path):
            user_folder_path = os.path.join(class_folder_path, user_folder)
            if os.path.isdir(user_folder_path):
                for img_file in os.listdir(user_folder_path):
                    if img_file.endswith(('png', 'jpg', 'jpeg')):  # Supported image formats
                        image_paths.append(os.path.join(user_folder_path, img_file))
                        labels.append(label)

# %%
#X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)





train_dataset = GreyscaleImageDataset(X_train, y_train, transform=transform)
val_dataset = GreyscaleImageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%
for i in train_loader:
    print(i[0].shape, i[1])
    break

# %%
model = MobileNet(num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
data = []

print("Training Started")

log_file_path = "../logs/side_logs.txt"
with open(log_file_path, "a") as log_file:
    log_file.write(", ".join(['Epoch', 'TLoss', "TAccuracy", 'VLoss', 'VAccuracy']))
    log_file.write("\n")
for epoch in range(epochs):
    d = []
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False):
        frames, labels = frames.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / total:.4f}, Accuracy: {100. * correct / total:.2f}%")
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"weights/Mobinet_{view}_{epoch + 1}_{learning_rate}.pth")
    d.extend([epoch + 1, train_loss / total, 100 * correct / total])
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation", leave=False):
            frames, labels = frames.cuda(), labels.cuda()
            outputs = model(frames)
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
        torch.save(model.state_dict(), weights_save_dir + f"Mobinet_{view}_best.pth")
        print(f"New best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.2f}%")
    
    # Save checkpoints, just in case?
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), weights_save_dir + f"Mobinet_{view}_{epoch + 1}_{learning_rate}.pth")

    # Final Save
    torch.save(model.state_dict(), weights_save_dir + f"Mobinet_{view}_final.pth")
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")
    os.system(f'echo "Best Side Weight {best_epoch}" > best_side_epoch.txt')




    print(f"Validation Loss: {val_loss / total:.4f}, Accuracy: {100. * correct / total:.2f}%")
    d.extend([val_loss / total, 100 * correct / total])
    with open(log_file_path, "a") as log_file:
        log_file.write(", ".join(map(str, d)))
        log_file.write("\n")
    # data.append(d)
    # print(d)


