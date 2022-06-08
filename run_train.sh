python3 main.py \
--category ab \
--workers 8 \
--batch_size 64 \
--epochs 230 \
--lr 1.5e-4 \
--encoder resnet34 \
--crop_size 320 \
--scheduler CosineAnnealingLR \
--lr_warmup_epochs 40 \
--lr_warmup_method linear \
--lr_warmup_decay 0.01 \
--weight_decay 1e-8 \
--criterion focal \
--val_epoch_start 170 \
