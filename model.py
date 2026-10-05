## import libraries
import os
import yaml
import torch
from ultralytics import YOLO

args = {}

args['batch_size'] = 2
args['epochs'] = 1
args['lr'] = 0.001
args['seed'] = 1
args['imgsz'] = 1280
args['data'] = './data/yolo/data.yaml'

if torch.cuda.is_available():
    args['device'] = 0
elif torch.backends.mps.is_available():
    args['device'] = 'mps'
else:
    args['device'] = 'cpu'


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
