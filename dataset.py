import os

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        category_map: dict,
        purpose: str,
        category: str,
        image_dirs: dict,
        augmentation=None,
        preprocessing=None,
        output_image_path: bool = False,
    ) -> None:

        self.category_map = category_map
        self.category = category
        self.image_dirs = image_dirs
        self.preprocessing = preprocessing
        self.augmentation = augmentation

        self.classes = category_map[category]
        self.class_values = [self.classes.index(cls.lower()) for cls in self.classes]

        satellite_suffixes = ["_bs", "_es", "_gs", "_ms", "_ys"]
        image_suffixes = [suf + ".tif" for suf in satellite_suffixes]
        mask_suffix = "_stacked.npy"
        mask_suffixv3 = "_stackedv3.npy"

        raw_imagefolders = list()
        with open(f"./dataset/{purpose}.txt") as file:
            raw_imagefolders = map(lambda dir: "../.." + dir.rstrip(), file.readlines())

        self.imagefolders = list()
        for imagefolder in raw_imagefolders:
            for area in self.image_dirs.keys():
                if area in imagefolder:
                    self.imagefolders.append(imagefolder)

        self.images, self.labels, self.imagepaths = [], [], []
        cities = set()

        for imagefolder in tqdm(self.imagefolders, desc=f"Preparing {purpose} dataset"):
            image_idx = imagefolder.split("/")[-2]

            for image_suffix in image_suffixes:
                imagepath = os.path.join(imagefolder, f"{image_idx}{image_suffix}")
                maskpath = os.path.join(imagefolder, f"{image_idx}{mask_suffix}")
                maskpathv3 = os.path.join(imagefolder, f"{image_idx}{mask_suffixv3}")

                if os.path.exists(imagepath) and os.path.exists(maskpathv3):
                    cities.add(imagepath.split("/")[-3])
                    self.images.append(imagepath)
                    self.labels.append(maskpathv3)
                elif os.path.exists(imagepath) and os.path.exists(maskpath):
                    self.images.append(imagepath)
                    self.labels.append(maskpath)
                self.imagepaths.append(imagefolder)

        print(cities, end="\n\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))

        category_index = list(self.category_map.keys()).index(self.category)

        mask = np.load(self.labels[idx])[category_index]

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask
