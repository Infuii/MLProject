#before preparing yolo you need to install yaml and pip install ultralytics

import json, os, random, yaml
from pathlib import Path
import shutil
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

def coco_to_yolo(json_file, img_dir, out_dir, val_frac=0.2, seed=1):
    coco = json.load(open(json_file))
    cat2idx = {c['id']: i for i, c in enumerate(coco['categories'])}  # 0-indexed, no background

    anns = {}
    for a in coco['annotations']:
        anns.setdefault(a['image_id'], []).append(a)

    existing = set(os.listdir(img_dir))
    images = [im for im in coco['images'] if im['file_name'] in existing]
    
    presence = np.zeros((len(images), len(cat2idx)), dtype='int8')
    for row, im in enumerate(images):
        for a in anns.get(im['id'], []):
            presence[row, cat2idx[a['category_id']]] = 1

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    _, val_idx = next(splitter.split(np.zeros((len(images), 1)), presence))
    val_set = set(val_idx.tolist())

    for sub in ('images', 'labels'):
        shutil.rmtree(Path(out_dir, sub), ignore_errors=True)

    for i, im in enumerate(images):
        split = 'val' if i in val_set else 'train'
        img_out = Path(out_dir, 'images', split)
        lbl_out = Path(out_dir, 'labels', split)
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        src = Path(img_dir) / im['file_name']
        dst = img_out / im['file_name']
        if not dst.exists():
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

        W, H = im['width'], im['height']
        lines = []

        for a in anns.get(im['id'], []):
            x, y, w, h = a['bbox']
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cat2idx[a['category_id']]} "
                         f"{(x + w/2)/W:.6f} {(y + h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")

        label_path = lbl_out / f"{Path(im['file_name']).stem}.txt"
        label_path.write_text("\n".join(lines))

    with open(Path(out_dir, 'data.yaml'), 'w') as f:
        yaml.safe_dump({
            'path': os.path.abspath(out_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: c['name'] for i, c in enumerate(coco['categories'])},
        }, f)
 
    print(f"Wrote {len(images) - len(val_set)} train / {len(val_set)} val images to {out_dir}")
 
 
if __name__ == '__main__':
    coco_to_yolo('./data/train_dataset.json', './data/train_images', './data/yolo')