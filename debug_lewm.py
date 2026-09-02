"""Diagnose what information a trained LeWM representation preserves."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mmnist import MovingMNIST
from utils import LeWM


def covariance_stats(features):
    features = features.float()
    centered = features - features.mean(0)
    cov = centered.T @ centered / (len(features) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    probabilities = eig / eig.sum()
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    off_diagonal = cov - cov.diag().diag()
    return {
        "mean_abs": features.mean(0).abs().mean().item(),
        "std_mean": features.std(0).mean().item(),
        "std_min": features.std(0).min().item(),
        "std_max": features.std(0).max().item(),
        "offdiag_rms": off_diagonal.square().mean().sqrt().item(),
        "effective_rank": effective_rank.item(),
    }


@torch.inference_mode()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    model = LeWM(
        cfg["latent_dim"],
        lamb=cfg["lamb"],
        slices=cfg["slices"],
        rollout_weight=cfg.get("rollout_weight", 1.0),
        sigreg_mode=cfg.get("sigreg_mode", "pooled"),
        action_dim=cfg.get("action_dim", 2),
    )
    model.load_state_dict(saved["model"])
    model.to(device).eval()

    dataset = MovingMNIST(args.data, False, args.samples, args.rollout, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    embeddings, labels, positions, trajectory_embeddings = [], [], [], []
    offset = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        b, t1 = frames.shape[:2]
        z = model.encoder(frames.flatten(0, 1)).view(b, t1, -1)
        embeddings.append(z[:, 0].cpu())
        trajectory_embeddings.append(z.cpu())
        indices = torch.arange(offset, offset + b)
        labels.append(dataset.mnist.targets[indices % len(dataset.mnist)])
        batch_positions = []
        for index in indices.tolist():
            generator = torch.Generator().manual_seed(dataset.seed + index)
            batch_positions.append(torch.randint(0, 41, (2,), generator=generator))
        positions.append(torch.stack(batch_positions))
        offset += b

    z0 = torch.cat(embeddings).numpy()
    all_z = torch.cat(trajectory_embeddings).flatten(0, 1)
    y = torch.cat(labels).numpy()
    xy = torch.cat(positions).numpy()
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=0, stratify=y
    )
    digit_probe = LogisticRegression(max_iter=500, n_jobs=-1).fit(z0[train], y[train])
    position_probe = Ridge(alpha=1.0).fit(z0[train], xy[train])
    stats = covariance_stats(all_z)

    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(
        f"SIGReg input: z={tuple(torch.cat(trajectory_embeddings).shape)} -> "
        f"flatten={tuple(all_z.shape)}; trajectories/batch={args.batch_size}, "
        f"frames/trajectory={args.rollout + 1}"
    )
    print(f"digit_linear_probe_acc={accuracy_score(y[test], digit_probe.predict(z0[test])):.4f}")
    print(f"position_linear_probe_r2={r2_score(xy[test], position_probe.predict(z0[test])):.4f}")
    print(" ".join(f"{key}={value:.4f}" for key, value in stats.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--data", default="data")
    main(parser.parse_args())
