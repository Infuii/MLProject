## import libraries
from __future__ import print_function
import argparse
import torch
import torchvision
import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

args = {}
kwargs = {}

args['batch_size'] = 2
args['test_batch_size'] = 2
args['epochs'] = 1
args['lr'] = 0.001
args['momentum'] = 0.5
args['seed'] = 1
args['log_interval'] = 10
args['cuda'] = torch.cuda.is_available()


class FathomNetDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            coco_data = json.load(f)


        #map FathomNet categories to 1-32.
        self.cat2idx = {cat['id']: i + 1 for i, cat in enumerate(coco_data['categories'])}
        self.num_classes = len(self.cat2idx) + 1  # +1 for background

        self.images_df = pd.DataFrame(coco_data['images'])
        self.annotations_df = pd.DataFrame(coco_data['annotations'])


        existing_files = set(os.listdir(img_dir))
        self.filtered_images = self.images_df[self.images_df['file_name'].isin(existing_files)].reset_index(drop=True)

        if len(self.filtered_images) == 0:
            raise RuntimeError(f"Could not find any images in {img_dir}.")

        print(f"Success: Found {len(self.filtered_images)} unique images ready for training.")

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.filtered_images)

    def __getitem__(self, idx):
        img_name = str(self.filtered_images.iloc[idx]['file_name'])
        img_id = self.filtered_images.iloc[idx]['id']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)


        img_annotations = self.annotations_df[self.annotations_df['image_id'] == img_id]

        boxes = []
        labels = []

        for _, row in img_annotations.iterrows():

            xmin = row['bbox'][0]
            ymin = row['bbox'][1]
            xmax = xmin + row['bbox'][2]
            ymax = ymin + row['bbox'][3]


            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat2idx[row['category_id']])


        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        return image, target



transform = transforms.Compose([
    transforms.ToTensor()
])


full_dataset = FathomNetDataset(json_file='./data/train_dataset.json',
                                img_dir='./data/train',
                                transform=transform)


total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size


if total_size < 2:
    print("Test Mode: Using the same image for train and validation.")
    train_split = full_dataset
    val_split = full_dataset
else:
    train_split, val_split = random_split(full_dataset, [train_size, val_size])
    print(f"Split complete: {train_size} training images, {val_size} validation images.")