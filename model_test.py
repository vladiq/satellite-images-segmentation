from argparse import ArgumentParser
import warnings

import matplotlib
import numpy as np
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import cv2

import utils_data

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
matplotlib.use("Agg")


parser = ArgumentParser()
parser.add_argument(
    "--category", type=str, required=True, help="Object category: aw, av, ab, ar"
)
parser.add_argument(
    "--workers",
    default=8,
    type=int,
    required=False,
    help="Number of data loading workers (default: 8)",
)
parser.add_argument(
    "--encoder",
    type=str,
    required=True,
    help="Encoder name from segmentation_models_pytorch",
)
parser.add_argument(
    "--decoder", type=str, required=True, help="Decoder name from smp in lowercase"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the model weights"
)
args = parser.parse_args()

palettes = {
    "av": [255, 255, 255]
    + [255, 255, 0]
    + [125, 125, 0]
    + [255, 150, 0]
    + [100, 150, 100],
    "ar": [255, 255, 255] + [30, 0, 0] + [255, 0, 0] + [50, 0, 0],
    "aw": [255, 255, 255] + [120, 120, 255] + [0, 0, 120] + [0, 0, 255],
    "ab": [255, 255, 255]
    + [0, 255, 0]
    + [130, 255, 200]
    + [0, 100, 0]
    + [100, 0, 100]
    + [128, 128, 128]
    + [0, 100, 255],
}

ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax2d"
DEVICE = "cuda"

if args.decoder == "deeplabv3plus":
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        classes=len(utils_data.category_map[args.category]),
        activation=None,
    )

elif args.decoder == "unet":
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        classes=len(utils_data.category_map[args.category]),
        activation=None,
    )

elif args.decoder == "fpn":
    model = smp.FPN(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        classes=len(utils_data.category_map[args.category]),
        activation=None,
    )

preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")
model.load_state_dict(torch.load(args.model_path))

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(DEVICE)

test_dataset = utils_data.CustomDataset(
    utils_data.category_map,
    purpose="val",
    category=args.category,
    image_dirs=utils_data.val_areas,
    # augmentation=transforms.get_validation_augmentation(),
    preprocessing=utils_data.get_preprocessing(preprocessing_fn),
)

test_dataset_vis = utils_data.CustomDataset(
    utils_data.category_map,
    purpose="val",
    category=args.category,
    image_dirs=utils_data.val_areas,
)

test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=args.workers)
print(f"Total test samples: {len(test_dataset)}")

phase = "inference"
data_loader = test_dataloader
num_classes = len(utils_data.category_map[args.category])

model.eval()
with torch.inference_mode():
    for idx in tqdm(range(len(test_dataset))):
        image_vis = test_dataset_vis[idx][0].astype("uint8")
        image, gt_mask = test_dataset[idx]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        image_vis = Image.fromarray(image_vis)
        image_vis.save(f"../results/{args.category}/{idx}_image.png")

        gt_mask = np.array(gt_mask).astype("uint8")
        gt_mask = Image.fromarray(gt_mask, mode="P")
        gt_mask.putpalette(palettes[args.category])
        gt_mask.save(f"../results/{args.category}/{idx}_mask_gt.png")

        pr_mask = cv2.resize(
            np.argmax(pr_mask, axis=0).astype("uint8"),
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST,
        )
        pr_mask = Image.fromarray(pr_mask, mode="P")
        pr_mask.putpalette(palettes[args.category])
        pr_mask.save(f"../results/{args.category}/{idx}_mask_pred.png")
