import webdataset as wds
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import io
import cv2
import glob
from torch.utils.data import DataLoader
from config import config


def get_augmentation(mode="train"):
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def decode_sample(sample):
    jpg_bytes = sample["jpg"]
    img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label_bytes = sample["labels.npy"]
    buf = io.BytesIO(label_bytes)
    labels = np.load(buf)
    labels = torch.tensor(labels, dtype=torch.float32)

    return img, labels


def apply_augmentation(sample, mode="train"):
    img, labels = sample
    transform = get_augmentation(mode)
    augmented = transform(image=img)
    img_tensor = augmented["image"]
    return img_tensor, labels


def get_dataloader(shard_pattern, mode="train"):
    shards = sorted(glob.glob(shard_pattern))