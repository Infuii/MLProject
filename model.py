## import libraries
import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import yaml
import torch
from ultralytics import YOLO
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

#-----------------------------------------------------------------
# Config
#-----------------------------------------------------------------

args = {
    'batch_size': 32,
    'epochs' : 5,
    'lr' : 0.001,
    'seed' : 1,
    'imgsz' : 1280,
    'val_frac': 0.2, #train/validation data split
    'freeze' : 11, #freeze YOLO backbone to only train head
    'train_json' : './data/train_dataset.json',
    'train_images' : './data/train_images',
    'data' : './data/yolo/data.yaml'
}

if torch.cuda.is_available():
    args['device'] = 0
elif torch.backends.mps.is_available():
    args['device'] = 'mps'
else:
    args['device'] = 'cpu'



#-----------------------------------------------------------------
# Data
#-----------------------------------------------------------------
def load_annotations(json_file, img_dir):
    # Return a DataFrame of (file_name, cat_id) rows for images
    # that exist on disk

    with open(json_file) as f:
        coco = json.load(f)

    merged = pd.merge(
        pd.DataFrame(coco['annotations']),
        pd.DataFrame(coco['images']),
        left_on='image_id',
        right_on='id'
    )

    existing = set(os.listdir(img_dir))
    df = merged[merged['file_name'].isin(existing)]

    return df[['file_name', 'category_id']].reset_index(drop=True)

def make_split_yaml(df):
    #split train/val by image, writes train.txt, val.txt, and data yaml

    labels_by_image = (
        df.groupby(['file_name', 'category_id']).size()
        .unstack(fill_value=0).gt(0).astype('int8')
    )
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args['val_frac'], random_state=args['seed']
    )
    train_idx, val_idx = next(splitter.split(
        np.zeros((len(labels_by_image), 1)), labels_by_image.to_numpy()
    ))


    def write_list(name, files):
        path = os.path.abspath(os.path.join(args['split_dir'], name))
        with open(path, 'w') as f:
            f.writelines(os.path.abspath(os.path.join(args['yolo_images'], fn)) + '\n' for fn in files)
        return path

    train_txt = write_list('train.txt', labels_by_image.index[train_idx])
    val_txt = write_list('val.txt', labels_by_image.index[val_idx])
    
    with open(args['base_data']) as f:
        cfg = yaml.safe_load(f)
    cfg.pop('path', None)  # train/val are absolute now
    cfg['train'], cfg['val'] = train_txt, val_txt
    
    out = os.path.join(args['split_dir'], 'data_split.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f)
    return out                        #transform=transform)


#-----------------------------------------------------------------
# Train / validate
#-----------------------------------------------------------------

if __name__ == '__main__':
    #data_yaml = make_split_yaml(load_annotations(args['train_json'], args['train_images']))

    model = YOLO('yolo11n.pt')
    model.train(
        data=args['data'],
        epochs=args['epochs'],
        batch=args['batch_size'],
        lr0=args['lr'],
        seed=args['seed'],
        imgsz=args['imgsz'],
        device=args['device'],
        freeze=args['freeze'],
    )

    metrics = model.val(data=args['data'], imgsz=args['imgsz'], device=args['device'])
    print(f"mAP50: {metrics.box.map50:.4f}   mAP50-95: {metrics.box.map:.4f}")


""" ## functions to show an image
def imshow(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean

    img = torch.clamp(img, 0, 1)

    
 # Test Mode: with only one image, prepare_yolo.py puts it in val and leaves train empty
with open(args['data']) as f:
    data_cfg = yaml.safe_load(f)
train_dir = os.path.join(data_cfg['path'], data_cfg['train'])
if not os.path.isdir(train_dir) or len(os.listdir(train_dir)) == 0:
    print("Test Mode: Using the same image for train and validation.")
    data_cfg['train'] = data_cfg['val']
    args['data'] = './data/yolo/data_testmode.yaml'
    with open(args['data'], 'w') as f:
        yaml.safe_dump(data_cfg, f) 


#Model
model = YOLO('yolo11n.pt')


#TRAINING
model.train(
    data=args['data'],
    epochs=args['epochs'],
    batch=args['batch_size'],
    lr0=args['lr'],
    seed=args['seed'],
    imgsz=args['imgsz'],
    device=args['device'],
)


#VALIDATION
metrics = model.val(data=args['data'], imgsz=args['imgsz'], device=args['device'])
print(f"mAP50: {metrics.box.map50:.4f}   mAP50-95: {metrics.box.map:.4f}")



 """

'''
class FathomNetDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        # 1. Open the FathomNet JSON file
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        # 2. Extract images and annotations into pandas DataFrames
        images_df = pd.DataFrame(coco_data['images'])
        annotations_df = pd.DataFrame(coco_data['annotations'])

        # 3. Merge them so every filename is matched with its category_id
        merged_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')

        # 4. CRITICAL FIX: Filter out missing images
        # Check the hard drive and ONLY keep rows in the JSON that match downloaded files
        existing_files = set(os.listdir(img_dir))
        filtered_df = merged_df[merged_df['file_name'].isin(existing_files)]

        if len(filtered_df) == 0:
            raise RuntimeError(f"Could not find any images in {img_dir} that match the JSON.")

        print(f"Success: Found {len(filtered_df)} downloaded image(s) ready for training.")

        # 5. Save the final mapped and filtered dataframe
        self.img_labels = filtered_df[['file_name', 'category_id']].reset_index(drop=True)
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


""" ## transformations (Updated for 3-channel RGB images!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Standard RGB normalization
]) """

'''