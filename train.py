import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from data import SegmentationDataset


DEVICE = "cuda"

EPOCHS = 100
LR = 0.002
IMAGE_SIZE = 320
BATCH_SIZE = 16

ENCODER = "timm-efficientnet-b0"
WEIGHTS = "imagenet"

CSV_FILE = "dataset.csv"
df = pd.read_csv(CSV_FILE)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)


def get_train_augs():
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    )


def get_val_augs():
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ]
    )


trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_val_augs())
