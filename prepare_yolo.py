#before preparing yolo you need to install yaml and pip install ultralytics

import json, os, random, yaml
from pathlib import Path

def coco_to_yolo(json_file, img_dir, out_dir, val_frac=0.2, seed=1):
    coco = json.load(open(json_file))
    cat2idx = {c['id']: i for i, c in enumerate(coco['categories'])}  # 0-indexed, no background

    anns = {}
    for a in coco['annotations']:
        anns.setdefault(a['image_id'], []).append(a)

    existing = set(os.listdir(img_dir))
    images = [im for im in coco['images'] if im['file_name'] in existing]
    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_frac))

    for i, im in enumerate(images):
        split = 'val' if i < n_val else 'train'
        img_out = Path(out_dir, 'images', split); img_out.mkdir(parents=True, exist_ok=True)
        lbl_out = Path(out_dir, 'labels', split); lbl_out.mkdir(parents=True, exist_ok=True)

        dst = img_out / im['file_name']
        if not dst.exists():
            os.symlink(os.path.abspath(os.path.join(img_dir, im['file_name'])), dst)

        W, H = im['width'], im['height']
        lines = []
        for a in anns.get(im['id'], []):
            x, y, w, h = a['bbox']
            if w <= 0 or h <= 0:          # same degenerate-box filter you have now
                continue
            lines.append(f"{cat2idx[a['category_id']]} "
                         f"{(x + w/2)/W:.6f} {(y + h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
        (lbl_out / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines))

    yaml.safe_dump({
        'path': os.path.abspath(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: c['name'] for i, c in enumerate(coco['categories'])},
    }, open(Path(out_dir, 'data.yaml'), 'w'))

coco_to_yolo('./data/train_dataset.json', './data/train', './data/yolo')
