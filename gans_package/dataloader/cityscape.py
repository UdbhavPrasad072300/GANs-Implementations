import os

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from PIL import Image


class CityScapeDataset(Dataset):
    def __init__(self, data_path="./data/cityscapes/"):
        super(CityScapeDataset, self).__init__()

        self.data_path = data_path
        self.data = {}

        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.feature_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        img = "_leftImg8bit.png"
        label = "_gtFine_labelIds.png"
        instance = "_gtFine_instanceIds.png"

        for root, dirs, files in os.walk(self.data_path):
            for f in files:
                if f.endswith(img):
                    name = f[:-len(img)]
                    if name not in self.data.keys():
                        self.data[name] = {}
                    self.data[name]["images"] = root + "/" + f
                elif f.endswith(label):
                    name = f[:-len(label)]
                    if name not in self.data.keys():
                        self.data[name] = {}
                    self.data[name]["labels"] = root + "/" + f
                elif f.endswith(instance):
                    name = f[:-len(instance)]
                    if name not in self.data.keys():
                        self.data[name] = {}
                    self.data[name]["instances"] = root + "/" + f

        self.data = list(self.data.values())

    def __getitem__(self, idx):
        image = self.image_transforms(Image.open(self.data[idx]["images"]))
        labels = self.image_transforms(Image.open(self.data[idx]["labels"]))
        instances = self.feature_transforms(Image.open(self.data[idx]["instances"]))
        return image, (labels, instances)

    def __len__(self):
        return len(self.data)


def getCityScapeDataLoader(data_path, batch_size=2, shuffle=True):
    return DataLoader(CityScapeDataset(data_path), batch_size=batch_size, shuffle=shuffle)
