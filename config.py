from pathlib import Path

class Config:
    # Data paths (HPC)
    SCRATCH        = Path("/scratch/taha8642")
    RAW_TRAIN      = SCRATCH / "data/train"
    RAW_VAL        = SCRATCH / "data/validation"
    SHARDS_TRAIN   = SCRATCH / "shards/train"
    SHARDS_VAL     = SCRATCH / "shards/val"
    TRAIN_CSV      = RAW_TRAIN / "SewerML_Train.csv"
    VAL_CSV        = RAW_VAL  / "SewerML_Val.csv"
    CHECKPOINT_DIR = SCRATCH / "checkpoints"

    # Classes — exact CSV column order, no WaterLevel, no Defect
    CLASSES = [
        "VA","RB","OB","PF","DE","FS","IS",
        "RO","IN","AF","BE","FO","GR","PH",
        "PB","OS","OP","OK","ND"
    ]
    NUM_CLASSES = 19

    # Image
    IMG_SIZE = 224

    # Training
    BATCH_SIZE     = 64
    NUM_EPOCHS     = 30
    LEARNING_RATE  = 1e-4
    NUM_WORKERS    = 4

config = Config()