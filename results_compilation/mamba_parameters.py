import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import cv2
import re
from mamba_ssm import Mamba
import torch.nn.functional as F


class MambaSequenceClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1364, num_classes=16):
        super().__init__()
        # self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.mamba_blockd = Mamba(
            d_model=input_dim,
            d_state=64,
            # d_conv=4,
            expand=2
        )
        self.mamba_blockr = Mamba(
            d_model=input_dim,
            d_state=64,
            # d_conv=4,
            expand=2
        )
        self.mamba_blocks = Mamba(
            d_model=input_dim,
            d_state=64,
            # d_conv=4,
            expand=2
        )
        # self.mamba_block.conv1d = nn.Identity()
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
    def forward(self, x, y, z):
        """
        x: [batch_size, seq_len, input_dim]
        returns: [batch_size, seq_len, num_classes]
        """ 
        # x = self.input_proj(x)  # -> [B, L, H]
        # print(x.shape)
        x = self.mamba_blockd(x)
        # print(y.shape, 'y')
        y = self.mamba_blockr(y)
        z = self.mamba_blocks(z)
        
        inp = torch.cat((x, y, z), 2)
        # print(inp.shape)
        x = self.fc1(inp)
        x = self.dropout(x)
        logits = self.output_layer(x)
        return logits


model = MambaSequenceClassifier(input_dim=2048, hidden_dim=2048, num_classes=16)

pytorch_total_params = sum(p.numel() for p in model.parameters())

pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total:", pytorch_total_params)
print("Trainable: ", pytorch_train_params)
