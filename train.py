import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
import numpy as np
import time
import os
from pathlib import Path

from config import config
from dataset import get_dataloader
from model import build_model


def compute_f1(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average='macro', zero_division=0)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    batches = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batches += 1

        if batches % 100 == 0:
            print(f"  Batch {batches}, Loss: {loss.item():.4f}")

    return total_loss / batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    batches = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())
            batches += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1 = compute_f1(all_preds, all_targets)

    return total_loss / batches, f1


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader = get_dataloader(
        '/scratch/taha8642/shards/train/shard-*.tar',
        mode='train'
    )
    val_loader = get_dataloader(
        '/scratch/taha8642/shards/val/shard-*.tar',
        mode='val'
    )

    # Model
    model = build_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    scaler = GradScaler()

    # Checkpoint dir
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Time: {elapsed/60:.1f}min")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, config.CHECKPOINT_DIR / 'swin_best.pth')
            print(f"  Saved best model with F1: {best_f1:.4f}")

        # Save last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
        }, config.CHECKPOINT_DIR / f'swin_epoch_{epoch+1}.pth')

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()