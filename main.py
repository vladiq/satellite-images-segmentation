import time
import datetime
from collections import defaultdict
import warnings

from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from torch.profiler import profile, record_function, ProfilerActivity

import utils
import utils_data

# torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)


def run_epoch(
    phase: str,
    category: str,
    num_classes: int,
    model: torch.nn.DataParallel,
    criterion,
    data_loader: torch.utils.data.DataLoader,
    optimizer=None,
    scheduler=None,
) -> dict:

    running_metrics = defaultdict(float)

    if phase == "train":
        model.train()
    else:
        model.eval()

    data_loader = tqdm(data_loader)
    data_loader.set_description(phase.capitalize())

    for batch_idx, (images, masks_gt) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        masks_gt = masks_gt.long().cuda(non_blocking=True)

        with torch.set_grad_enabled(phase == "train"):
            masks_pred = model(images)
            loss = criterion(masks_pred, masks_gt)

            if phase == "train":
                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        cur_metrics = utils.get_running_metrics(
            phase, masks_pred, masks_gt, category, num_classes
        )

        running_metrics["mean_loss"] += loss.item()
        for metric_name, val in cur_metrics.items():
            running_metrics[metric_name] += val

        if batch_idx == len(data_loader) - 1:
            data_loader.set_postfix(
                {
                    metric: f"{(val / len(data_loader)):.4f}"
                    for metric, val in running_metrics.items()
                }
            )

    if phase == "train":
        scheduler.step()

    for metric in running_metrics.keys():
        running_metrics[metric] /= len(data_loader)

    return running_metrics


def main():
    args = utils.get_argparser().parse_args()
    print(args)

    num_classes = len(utils_data.category_map[args.category])

    model = smp.FPN(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None,
    )

    # model.load_state_dict(torch.load('./models/av_best_model_val_dice_0.13899.pth'))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")

    train_dataset = utils_data.CustomDataset(
        utils_data.category_map,
        purpose="train",
        category=args.category,
        image_dirs=utils_data.train_areas,
        augmentation=utils_data.get_training_augmentation(crop_size=args.crop_size),
        preprocessing=utils_data.get_preprocessing(preprocessing_fn),
    )
    valid_dataset = utils_data.CustomDataset(
        utils_data.category_map,
        purpose="val",
        category=args.category,
        image_dirs=utils_data.val_areas,
        augmentation=utils_data.get_validation_augmentation(),
        preprocessing=utils_data.get_preprocessing(preprocessing_fn),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    criterion = utils.get_criterion(args.criterion)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = utils.get_lr_scheduler(len(train_loader), optimizer, args)

    start_time = time.time()
    max_valid_miou = float("-inf")

    learning_rates = list()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")

        # learning_rates.append(optimizer.param_groups[0]["lr"])

        train_metrics = run_epoch(
            "train",
            args.category,
            num_classes,
            model,
            criterion,
            train_loader,
            optimizer,
            scheduler,
        )

        if epoch % 3 == 0 and epoch > args.val_epoch_start:
            valid_metrics = run_epoch(
                "valid", args.category, num_classes, model, criterion, valid_loader
            )

            mean_iou_score = valid_metrics["mean_iou_score"]

            if mean_iou_score > max_valid_miou:
                max_valid_miou = mean_iou_score

                if torch.cuda.device_count() > 1:
                    torch.save(
                        model.module.state_dict(),
                        f"../models/{args.category}_{args.criterion}_max_val_miou_{max_valid_miou:.5f}_{args.encoder}.pth",
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        f"../models/{args.category}_{args.criterion}_max_val_miou_{max_valid_miou:.5f}_{args.encoder}.pth",
                    )

                print("\t\t\tMODEL SAVED")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # print(learning_rates)


if __name__ == "__main__":
    main()
