import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.images
        masks_path = row.masks

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = np.expand_dims(mask, axis=-1)  # h,w,c

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]

        image = np.transpose(image, (2, 0, 1)).astype(float)
        mask = np.transpose(mask, (2, 0, 1)).astype(float)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask))

        return (image, mask)
