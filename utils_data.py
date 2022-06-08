import os

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import albumentations as A

areas = [
    "antwerpen",
    "barcelona",
    "bergheim",
    "bristol",
    "brussel",
    "budapest",
    "dubai_cleaned",
    "eschweiler",
    "isab_erg",
    "kadan",
    "kerch",
    "kochenevo",
    "lisbon",
    "madrid",
    "madrid1",
    "madrid2",
    "manila",
    "Moscow",
    "munich",
    "north_yorkshire",
    "novopolozk",
    "npz",
    "pavlodar",
    "prague",
    "rogowiec",
    "rotterdam",
    "sahara",
    "scandinavian_mountains_clean",
    "sevastopol",
    "singapoore",
    "spb",
    "surgut",
    "swierze_gorne",
    "tumen",
    "vena",
]

train_areas = {area: f"../../../stordb/geosegmentation/mis/{area}" for area in areas}
val_areas = {area: f"../../../stordb/geosegmentation/mis/{area}" for area in areas}

category_map = {
    "aw": ["background", "river_channel", "pond", "ocean_sea_lake"],
    "av": ["background", "low_vegetation", "woodland", "sand", "mountains"],
    "ab": [
        "background",
        "building",
        "seaport",
        "powerstation",
        "industrial_building",
        "npz",
        "airport",
    ],
    "ar": ["background", "bridge", "road", "railway"],
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        category_map: dict,
        purpose: str,
        category: str,
        image_dirs: dict,
        augmentation=None,
        preprocessing=None,
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
        with open(f"./data_paths/{purpose}.txt") as file:
            raw_imagefolders = map(
                lambda dir: "../../.." + dir.rstrip(), file.readlines()
            )

        self.imagefolders = list()
        for imagefolder in raw_imagefolders:
            for area in self.image_dirs.keys():
                if area in imagefolder:
                    self.imagefolders.append(imagefolder)

        self.images, self.labels = [], []
        cities = set()

        for idx, imagefolder in enumerate(
            tqdm(self.imagefolders, desc=f"Preparing {purpose} dataset")
        ):
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
                    cities.add(imagepath.split("/")[-3])
                    self.images.append(imagepath)
                    self.labels.append(maskpath)

        if purpose == "val":
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


def get_training_augmentation(crop_size=320):
    train_transform = [
        A.OneOf(
            [
                A.RandomCrop(crop_size, crop_size, p=1),
                A.RandomResizedCrop(crop_size, crop_size, p=1, ratio=(0.5, 2)),
            ],
            p=1,
        ),
        A.Flip(),
        A.RandomBrightnessContrast(),
        # A.CenterCrop(320, 320, p=1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        # A.Resize(320, 320, always_apply=True)
    ]
    return A.Compose(test_transform)


def to_tensor_1(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def to_tensor_2(x, **kwargs):
    return x.astype("int32")


def get_preprocessing(preprocessing_fn):
    return A.Compose(
        [
            A.Lambda(image=preprocessing_fn),
            A.Lambda(image=to_tensor_1, mask=to_tensor_2),
        ]
    )
