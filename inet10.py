"""Fast LeJEPA signal on 64x64 ImageNette with an online linear probe."""

import argparse
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from utils import LeJEPA


def loader(split, batch_size, workers, views=1):
    def crop():
        return transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((64, 64), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        )
    transform = transforms.MultiViewTransform({f"view_{i}": crop() for i in range(views)})
    dataset = spt.data.HFDataset(
        "frgfm/imagenette",
        "320px",
        split=split,
        transform=transform,
        trust_remote_code=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=split == "train",
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def main(args):
    args.output = str(Path(args.output).expanduser().resolve())
    spt.set(cache_dir=args.output, requeue_checkpoint=False, verbose="WARNING")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    train = loader("train", args.batch_size, args.workers, args.views)
    val = loader("validation", args.batch_size * 2, args.workers)
    lejepa = LeJEPA(lamb=args.lamb, slices=args.slices)
    if torch.cuda.is_available():
        lejepa.encoder = torch.compile(lejepa.encoder, mode="reduce-overhead")
        lejepa.projector = torch.compile(lejepa.projector, mode="reduce-overhead")

    def forward(self, batch, stage):
        if stage == "fit":
            views = [batch[k] for k in sorted(k for k in batch if k.startswith("view_"))]
            images = torch.stack([view["image"] for view in views], 1)
            loss, inv, sig, embedding = self.model(images)
            embedding, label = embedding[0], views[0]["label"].long()
            self.log_dict(
                {"train/lejepa": loss, "train/invariance": inv, "train/sigreg": sig}, on_epoch=True
            )
        else:
            view = batch["view_0"]
            embedding = self.model.encoder(view["image"])
            label, loss = view["label"].long(), embedding.sum() * 0
        return {"loss": loss, "embedding": embedding, "label": label}

    module = spt.Module(
        model=lejepa,
        forward=forward,
        hparams=vars(args),
        optim={
            "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 5e-2},
        },
    )
    probe = spt.callbacks.OnlineProbe(
        module,
        "probe",
        "embedding",
        "label",
        nn.Linear(512, 10),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(10)},
    )
    logger = CSVLogger(args.output, name="inet10")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if args.steps > 0 else args.epochs,
        limit_train_batches=args.steps if args.steps > 0 else 1.0,
        limit_val_batches=4 if args.steps > 0 else 1.0,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[probe],
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        seed=args.seed,
    )
    manager()
    score = float(trainer.callback_metrics["eval/probe_acc"])
    print(f"RESULT inet10 online_probe_acc={score:.4f} logs={manager._run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=29)
    parser.add_argument("--steps", type=int, default=-1, help="Override epochs for a smoke run")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.8e-3)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--lamb", type=float, default=0.01)
    parser.add_argument("--slices", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs")
    main(parser.parse_args())
