"""Action-conditioned LeWM on Moving-MNIST, including a decoded rollout GIF."""

import argparse
from pathlib import Path

import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

import stable_pretraining as spt
from utils import LeWM, save_rollout, save_rollout_grid


class MovingMNIST(Dataset):
    """One MNIST digit translated by the observed action (Δx, Δy)."""

    def __init__(
        self, root="data", train=True, size=10_000, steps=8, seed=0, rotation_step=0
    ):
        self.mnist = MNIST(root, train=train, download=True)
        self.size, self.steps, self.seed = size, steps, seed
        self.rotation_step = rotation_step

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(self.seed + index)
        digit = self.mnist.data[index % len(self.mnist)].float().div_(255)[None, None]
        digit = F.interpolate(digit, (24, 24), mode="bilinear", align_corners=False)[0]
        pos = torch.randint(0, 41, (2,), generator=g)
        angle = 0
        frames, actions, positions = [], [], []
        for t in range(self.steps + 1):
            frame = torch.zeros(1, 64, 64)
            x, y = pos.tolist()
            visible_digit = rotate(
                digit,
                angle=float(angle),
                interpolation=InterpolationMode.BILINEAR,
            )
            frame[:, y : y + 24, x : x + 24] = visible_digit
            frames.append(frame)
            positions.append(pos.float() / 40)
            if t < self.steps:
                requested = torch.randint(-4, 5, (2,), generator=g)
                new_pos = (pos + requested).clamp(0, 40)
                action = [(new_pos - pos).float() / 4]
                if self.rotation_step:
                    rotation_action = torch.randint(-1, 2, (), generator=g)
                    action.append(rotation_action.float()[None])
                    angle = (angle + int(rotation_action) * self.rotation_step) % 360
                actions.append(torch.cat(action))
                pos = new_pos
        return {
            "frames": torch.stack(frames),
            "actions": torch.stack(actions),
            "digit": self.mnist.targets[index % len(self.mnist)].long(),
            "positions": torch.stack(positions),
        }


def main(args):
    args.output = str(Path(args.output).expanduser().resolve())
    spt.set(cache_dir=args.output, requeue_checkpoint=False, verbose="WARNING")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    train_ds = MovingMNIST(
        args.data, True, args.samples, args.rollout, args.seed, args.rotation_step
    )
    val_size = min(512, max(args.batch_size * 2, args.samples // 4))
    val_ds = MovingMNIST(
        args.data,
        False,
        val_size,
        args.rollout,
        args.seed + 1_000_000,
        args.rotation_step,
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            drop_last=shuffle,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    world_model = LeWM(
        args.latent_dim,
        lamb=args.lamb,
        slices=args.slices,
        rollout_weight=args.rollout_weight,
        sigreg_mode=args.sigreg_mode,
        action_dim=3 if args.rotation_step else 2,
    )
    if args.compile and torch.cuda.is_available():
        world_model.encoder = torch.compile(world_model.encoder, mode="reduce-overhead")

    def forward(self, batch, stage):
        loss, pred, sig, z = self.model(batch["frames"], batch["actions"])
        self.log_dict(
            {f"{stage}/lewm": loss, f"{stage}/prediction": pred, f"{stage}/sigreg": sig},
            on_epoch=True,
        )
        return {
            "loss": loss,
            "embedding": z.flatten(0, 1),
            "decode_target": batch["frames"].flatten(0, 1),
            "digit_target": batch["digit"][:, None].expand(-1, z.shape[1]).flatten(),
            "position_target": batch["positions"].flatten(0, 1),
        }

    module = spt.Module(
        model=world_model,
        forward=forward,
        hparams=vars(args),
        optim={
            "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-4},
            "scheduler": {"type": "CosineAnnealingLR"},
            "interval": "step",
        },
    )
    digit_probe = spt.callbacks.OnlineProbe(
        module,
        "digit_probe",
        "embedding",
        "digit_target",
        nn.Linear(args.latent_dim, 10),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(10)},
    )
    position_probe = spt.callbacks.OnlineProbe(
        module,
        "position_probe",
        "embedding",
        "position_target",
        nn.Linear(args.latent_dim, 2),
        loss=nn.MSELoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={
            "r2": torchmetrics.regression.R2Score(multioutput="variance_weighted")
        },
    )
    digit_mlp_probe = spt.callbacks.OnlineProbe(
        module,
        "digit_mlp_probe",
        "embedding",
        "digit_target",
        nn.Sequential(
            nn.Linear(args.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 10),
        ),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(10)},
    )
    position_mlp_probe = spt.callbacks.OnlineProbe(
        module,
        "position_mlp_probe",
        "embedding",
        "position_target",
        nn.Sequential(
            nn.Linear(args.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        ),
        loss=nn.MSELoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={
            "r2": torchmetrics.regression.R2Score(multioutput="variance_weighted")
        },
    )
    decoder_loss = None
    if args.decoder_foreground_weight > 0:
        foreground_weight = args.decoder_foreground_weight

        def decoder_loss(prediction, target):
            weights = 1 + foreground_weight * target
            return (weights * (prediction - target).square()).sum() / weights.sum()

    decoder = spt.callbacks.OnlineImageDecoder(
        module,
        "decoder",
        "embedding",
        "decode_target",
        (1, 64, 64),
        args.latent_dim,
        decoder_kwargs={"base_channels": 128, "min_channels": 16, "num_res_blocks": 1},
        loss=decoder_loss,
        optimizer={"type": "Adam", "lr": 1e-3},
    )
    logger = CSVLogger(args.output, name="mmnist")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if args.steps > 0 else args.epochs,
        limit_train_batches=args.steps if args.steps > 0 else 1.0,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            decoder,
            digit_probe,
            position_probe,
            digit_mlp_probe,
            position_mlp_probe,
        ],
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=make_loader(train_ds, True), val=make_loader(val_ds, False)),
        seed=args.seed,
    )
    manager()

    checkpoint = Path(manager._run_dir) / "lewm.pt"
    model_state = {
        key.replace("encoder._orig_mod.", "encoder.").replace(
            "predictor._orig_mod.", "predictor."
        ): value
        for key, value in world_model.state_dict().items()
    }
    torch.save(
        {
            "model": model_state,
            "decoder": module.callbacks_modules["decoder"].state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "lamb": args.lamb,
                "slices": args.slices,
                "rollout_weight": args.rollout_weight,
                "sigreg_mode": args.sigreg_mode,
                "decoder_foreground_weight": args.decoder_foreground_weight,
                "action_dim": 3 if args.rotation_step else 2,
                "rotation_step": args.rotation_step,
            },
        },
        checkpoint,
    )
    indices = [int((val_ds.mnist.targets == digit).nonzero()[0]) for digit in range(10)]
    samples = [val_ds[index] for index in indices]
    device = module.device
    with torch.inference_mode():
        z = world_model.rollout(
            torch.stack([sample["frames"][0] for sample in samples]).to(device),
            torch.stack([sample["actions"] for sample in samples]).to(device),
        )
        decoded = module.callbacks_modules["decoder"](z.flatten(0, 1)).float().cpu()
        decoded = decoded.view(10, args.rollout + 1, 1, 64, 64)
    truths = torch.stack([sample["frames"] for sample in samples])
    actions = torch.stack([sample["actions"] for sample in samples])
    for digit in range(10):
        save_rollout(
            Path(manager._run_dir) / f"rollout_{digit}.gif",
            truths[digit],
            decoded[digit],
            actions[digit],
            args.fps,
        )
    gif = save_rollout_grid(
        Path(manager._run_dir) / "rollout.gif", truths, decoded, actions, range(10), args.fps
    )
    pred_mse = float(trainer.callback_metrics["validate/prediction"])
    digit_acc = float(trainer.callback_metrics["eval/digit_probe_acc"])
    position_r2 = float(trainer.callback_metrics["eval/position_probe_r2"])
    digit_mlp_acc = float(trainer.callback_metrics["eval/digit_mlp_probe_acc"])
    position_mlp_r2 = float(trainer.callback_metrics["eval/position_mlp_probe_r2"])
    print(
        f"RESULT mmnist digit_linear_acc={digit_acc:.4f} "
        f"digit_mlp_acc={digit_mlp_acc:.4f} position_linear_r2={position_r2:.4f} "
        f"position_mlp_r2={position_mlp_r2:.4f} "
        f"rollout_prediction_mse={pred_mse:.6f} video={gif} checkpoint={checkpoint}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=-1, help="Override epochs for a smoke run")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=8)
    parser.add_argument(
        "--rotation-step",
        type=int,
        default=0,
        help="Add a third rotation action with -1/0/+1 steps of this many degrees",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument(
        "--sigreg-mode",
        choices=("pooled", "per_time", "both", "pooled_pred"),
        default="pooled",
    )
    parser.add_argument("--slices", type=int, default=2048)
    parser.add_argument(
        "--decoder-foreground-weight",
        type=float,
        default=0.0,
        help="Weight bright digit pixels in the online decoder's reconstruction MSE",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="runs")
    main(parser.parse_args())
