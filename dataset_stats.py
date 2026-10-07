import json, pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

with open (f"train_dataset.json") as f:
    data = json.load(f)

print (data.keys())

images = pd.DataFrame(data["images"])
anns   = pd.DataFrame(data["annotations"])
cats   = pd.DataFrame(data["categories"])

anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)

df = (anns
      .merge(images, left_on="image_id", right_on="id", suffixes=("", "_img"))
      .merge(cats[["id", "name"]], left_on="category_id", right_on="id",
             suffixes=("", "_cat")))

#printing summary stats
print (len(images), "images;", len(anns), "annotations; ", len(cats), "classes.")

class_counts = df["name"].value_counts()
print(class_counts.describe())
print(class_counts.head(10)); print (class_counts.tail(10))
print(class_counts.to_string())

def show_random_grid(n=9, cols=3, seed=None, class_name=None, figsize_per=(6, 4.5)):
    """
    Show a random grid of images with bounding boxes drawn on top.

    n           : number of images to show
    cols        : number of columns in the grid
    seed        : set for reproducible samples
    class_name  : if given, only sample images containing that class
    """
    pool = df if class_name is None else df[df["name"] == class_name]

    # Map each image_id to its file_name, then keep only images that exist on disk
    id_to_file = pool.drop_duplicates("image_id").set_index("image_id")["file_name"]
    existing_mask = id_to_file.apply(lambda fn: os.path.exists(f"images/{fn}"))
    valid_ids = id_to_file[existing_mask].index

    missing_count = len(id_to_file) - len(valid_ids)
    if missing_count > 0:
        print(f"Skipping {missing_count} image(s) with missing files.")

    if len(valid_ids) == 0:
        print("No valid images found to display.")
        return

    ids = pd.Series(valid_ids).sample(min(n, len(valid_ids)), random_state=seed).tolist()

    rows_n = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(figsize_per[0] * cols, figsize_per[1] * rows_n))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    # One consistent color per class
    cmap = plt.get_cmap("tab20")
    names = sorted(df["name"].unique())
    colors = {nm: cmap(i % 20) for i, nm in enumerate(names)}

    for ax, image_id in zip(axes, ids):
        rows = df[df["image_id"] == image_id]
        fn = rows["file_name"].iloc[0]
        img_path = f"images/{fn}"

        # Defensive check in case the file disappears between filtering and plotting
        if not os.path.exists(img_path):
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10)
            ax.set_title(f"id {image_id} (missing file)", fontsize=9)
            ax.axis("off")
            continue

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)

        for _, r in rows.iterrows():
            c = colors[r["name"]]
            ax.add_patch(patches.Rectangle((r["x"], r["y"]), r["w"], r["h"],
                                           linewidth=2, edgecolor=c, facecolor="none"))
            ax.text(r["x"], max(r["y"] - 4, 0), r["name"], color="white", fontsize=8,
                    bbox=dict(facecolor=c, alpha=0.8, pad=1, edgecolor="none"))

        ax.set_title(f"id {image_id} ({len(rows)} boxes)", fontsize=9)
        ax.axis("off")

    # Hide any unused axes
    for ax in axes[len(ids):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

show_random_grid()