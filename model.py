import torch
import torch.nn as nn
import timm
from config import config


def build_model(num_classes=config.NUM_CLASSES, pretrained=True):
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_f1 = checkpoint.get('best_f1', 0)
    print(f"Loaded checkpoint: epoch {epoch}, best F1: {best_f1:.4f}")
    return model, epoch, best_f1


if __name__ == "__main__":
    model = build_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")