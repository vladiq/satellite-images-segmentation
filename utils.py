from argparse import ArgumentParser

import torch
import segmentation_models_pytorch as smp

import utils_data


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--category", type=str, required=True, help="Object category: aw, av, ab, ar"
    )
    parser.add_argument(
        "--workers",
        default=16,
        type=int,
        required=False,
        help="Number of data loading workers (default: 16)",
    )
    parser.add_argument(
        "--val_epoch_start",
        type=int,
        help="Epoch number since which the validation loop will start working",
    )

    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Encoder name from segmentation_models_pytorch",
    )
    parser.add_argument(
        "--crop_size", type=int, required=True, help="Crop size for training"
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay (default:1e-8)"
    )

    parser.add_argument(
        "--scheduler", type=str, required=True, help="Main LR scheduler"
    )
    parser.add_argument(
        "--lr_warmup_epochs", type=int, required=True, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--lr_warmup_method",
        type=str,
        default="linear",
        help="constant or linear (default: linear)",
    )
    parser.add_argument(
        "--lr_warmup_decay",
        type=float,
        default=0.01,
        help="The decay for lr (default: 0.01)",
    )

    parser.add_argument(
        "--criterion", type=str, help="Loss function: dice, jaccard, focal"
    )

    return parser


def get_lr_scheduler(iters_per_epoch, optimizer, args):
    if args.scheduler == "CosineAnnealingLR":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, verbose=True
        )
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-7, verbose=True
        )
    elif args.scheduler == "LambdaLR":
        main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (
                1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))
            )
            ** 9,
            verbose=True,
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
                verbose=True,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
                verbose=True,
            )
        else:
            raise RuntimeError("Invalid lr warmup method. Use linear or constant")

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
            verbose=True,
        )

    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler


def get_criterion(criterion_name: str):
    if criterion_name == "dice":
        criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    elif criterion_name == "focal":
        criterion = smp.losses.FocalLoss(mode="multiclass")
    elif criterion_name == "jaccard":
        criterion = smp.losses.JaccardLoss(mode="multiclass", from_logits=True)
    else:
        raise RuntimeError("Invalid criterion name. Use dice, focal or jaccard")
    return criterion


@torch.no_grad()
def get_running_metrics(phase, masks_pred, masks_gt, category, num_classes) -> dict():
    results = dict()

    masks_pred = torch.argmax(masks_pred, dim=1).long()

    tp, fp, fn, tn = smp.metrics.get_stats(
        masks_pred, masks_gt, mode="multiclass", num_classes=num_classes
    )

    results["mean_iou_score"] = smp.metrics.iou_score(
        tp, fp, fn, tn, reduction="micro"
    ).item()

    if phase == "inference":
        results["f1_score"] = smp.metrics.f1_score(
            tp, fp, fn, tn, reduction="micro"
        ).item()
        results["pixelwise_acc"] = smp.metrics.accuracy(
            tp, fp, fn, tn, reduction="micro"
        ).item()
        results["classwise_acc"] = smp.metrics.accuracy(
            tp, fp, fn, tn, reduction="macro"
        ).item()

        masks_gt_onehot = torch.nn.functional.one_hot(masks_gt, num_classes=num_classes)
        masks_pred_onehot = torch.nn.functional.one_hot(
            masks_pred, num_classes=num_classes
        )

        for class_idx, class_name in enumerate(utils_data.category_map[category]):
            class_mask_pred = masks_pred_onehot[..., class_idx]
            class_mask_gt = masks_gt_onehot[..., class_idx]

            tp, fp, fn, tn = smp.metrics.get_stats(
                class_mask_pred, class_mask_gt, mode="binary"
            )
            results[f"{class_name}_iou"] = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro"
            ).item()

    return results
