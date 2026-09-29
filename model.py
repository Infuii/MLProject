## import libraries
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

args={}
kwargs={}
args['batch_size']=32
args['test_batch_size']=32
args['epochs']=1  #The number of Epochs is the number of times you go through the full dataset.
args['lr']=0.001 #Learning rate is how fast it will decend.
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=False #if the computer has a GPU, type True, otherwise, False


class FathomNetDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        # 1. Open the FathomNet JSON file
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        # 2. Extract images and annotations into pandas DataFrames
        images_df = pd.DataFrame(coco_data['images'])
        annotations_df = pd.DataFrame(coco_data['annotations'])

        # 3. Merge them so every filename is matched with its category_id
        # In COCO JSON, images use 'id' and annotations use 'image_id'
        merged_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')

        # 4. Save the final mapping
        self.img_labels = merged_df[['file_name', 'category_id']]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(self.img_labels.iloc[idx]['file_name'])
        img_path = os.path.join(self.img_dir, img_name)


        image = Image.open(img_path).convert("RGB")

        # Get integer label
        label = int(self.img_labels.iloc[idx]['category_id'])

        if self.transform:
            image = self.transform(image)

        return image, label

## transformations (Updated for 3-channel RGB images!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Standard RGB normalization
])

## download and load training dataset
trainset = FathomNetDataset(json_file='./data/train_dataset.json',
                            img_dir='./data/train',
                            transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, **kwargs)

## download and load testing dataset
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'], shuffle=True, **kwargs)

## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

## get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

## show images
imshow(torchvision.utils.make_grid(images))