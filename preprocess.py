"""
preprocess.py
Reads raw PNGs/JPGs + CSV, resizes to 224x224, writes WebDataset tar shards.
Run once on HPC. Never again.
"""
import webdataset as wds
import pandas as pd
import numpy as np
import cv2
import io
import random
from pathlib import Path
from config import config

SHARD_SIZE = 1000

def preprocess(split="train"):
    if split == "train":
        csv_path = config.TRAIN_CSV
        img_dirs = [
            config.RAW_TRAIN / "train00",
            config.RAW_TRAIN / "train01",
            config.RAW_TRAIN / "train02",
            config.RAW_TRAIN / "train03",
        ]
        out_dir = config.SHARDS_TRAIN
    else:
        csv_path = config.VAL_CSV
        img_dirs = [config.RAW_VAL / "valid01"]
        out_dir = config.SHARDS_VAL

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {len(df)} rows")

    label_cols = config.CLASSES
    df = df.set_index("Filename")

    # Build image path lookup — handles both png and jpg
    img_lookup = {}
    for d in img_dirs:
        for ext in ["*.png", "*.jpg"]:
            for p in Path(d).glob(ext):
                img_lookup[p.name] = p

    print(f"Images found: {len(img_lookup)}")

    valid = [f for f in df.index if f in img_lookup]
    print(f"Valid samples: {len(valid)}")

    # Shuffle so shards aren't ordered by folder
    random.shuffle(valid)

    shard_pattern = str(out_dir / "shard-%06d.tar")
    total = 0
    skipped = 0

    with wds.ShardWriter(shard_pattern, maxcount=SHARD_SIZE) as sink:
        for filename in valid:
            img_path = img_lookup[filename]

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_bytes = buf.tobytes()

            labels = df.loc[filename, label_cols].values.astype("float32")

            # Save labels with shape/dtype preserved
            label_buf = io.BytesIO()
            np.save(label_buf, labels)
            label_bytes = label_buf.getvalue()

            sink.write({
                "__key__": filename.split(".")[0],
                "jpg": jpg_bytes,
                "labels.npy": label_bytes,
            })
            total += 1

            if total % 10000 == 0:
                print(f"Processed {total}/{len(valid)}")

    print(f"Done. Total: {total}, Skipped: {skipped}")
    print(f"Shards written to: {out_dir}")

if __name__ == "__main__":
    print("=== Preprocessing TRAIN ===")
    preprocess("train")
    print("=== Preprocessing VAL ===")
    preprocess("val")