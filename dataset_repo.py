from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CovidDataset(Dataset):
    def __init__(self, data_dict, transform=None) -> None:
        super().__init__()
        self.root_data_path = Path(data_dict["data dir"])
        self._label_names = data_dict["label names"]
        self.image_size = tuple(data_dict["image size"][1:])
        self.transform = transform
        self.imgPath_lbl = []

    def __len__(self):
        return len(self.imgPath_lbl)

    def __getitem__(self, idx):
        image_path = self.imgPath_lbl[idx][0]
        label_name = self.imgPath_lbl[idx][1]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "image_path": str(image_path),
            "label": str(label_name),
            "class": torch.tensor(self.imgPath_lbl[idx][2], dtype=torch.uint8),
        }


class Covid_Xray(CovidDataset):
    def __init__(self, data_dict, image_size) -> None:
        super().__init__(data_dict)
        self.imgPath_lbl = [
            (image_path, name, label)
            for label, name in enumerate(self._label_names)
            for image_path in self.root_data_path.joinpath(name).rglob("*.png")
        ]


class Covid_QU_Ex(CovidDataset):
    def __init__(self, data_dict, mode="train", transform=None) -> None:
        super().__init__(data_dict, transform)

        if mode == "train":
            self.root_data_path = self.root_data_path.joinpath("Train")
        elif mode == "val":
            self.root_data_path = self.root_data_path.joinpath("Val")
        elif mode == "test":
            self.root_data_path = self.root_data_path.joinpath("Test")
        else:
            raise ValueError("mode must be train or val or test")

        self.imgPath_lbl = [
            (image_path, name, label)
            for label, name in enumerate(self._label_names)
            for image_path in self.root_data_path.joinpath(name, "images").rglob(
                "*.png"
            )
        ]
