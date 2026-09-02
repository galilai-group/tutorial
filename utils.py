"""Tiny LeJEPA/LeWM building blocks shared by the tutorials."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn

import stable_pretraining as spt
from stable_pretraining.methods.lejepa import SlicedEppsPulley


class LeJEPA(nn.Module):
    """ResNet9 + projector + invariance/SIGReg objective."""

    def __init__(self, embed_dim=512, proj_dim=128, lamb=0.01, slices=256):
        super().__init__()
        self.encoder = spt.backbone.Resnet9(embed_dim, 3)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim),
        )
        self.sigreg = SlicedEppsPulley(num_slices=slices)
        self.lamb = lamb

    def forward(self, images):
        """Return loss parts and embeddings; ``images`` is [B,V,C,H,W]."""
        b, v = images.shape[:2]
        embedding = self.encoder(images.flatten(0, 1)).view(b, v, -1).transpose(0, 1)
        projection = self.projector(embedding.flatten(0, 1)).view(v, b, -1)
        invariance = (projection - projection.mean(0)).square().mean()
        sigreg = self.sigreg(projection.flatten(0, 1))
        return invariance + self.lamb * sigreg, invariance, sigreg, embedding


class LeWM(nn.Module):
    """Small action-conditioned latent world model for 64x64 grayscale video."""

    def __init__(
        self,
        latent_dim=64,
        hidden_dim=256,
        lamb=0.1,
        slices=256,
        rollout_weight=1.0,
        sigreg_mode="pooled",
        action_dim=2,
    ):
        super().__init__()
        self.encoder = spt.backbone.Resnet9(latent_dim, 1)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.sigreg = SlicedEppsPulley(num_slices=slices)
        self.lamb = lamb
        self.rollout_weight = rollout_weight
        self.sigreg_mode = sigreg_mode
        self.action_dim = action_dim

    def step(self, latent, action):
        return self.predictor(torch.cat((latent, action), -1))

    def forward(self, frames, actions):
        """Teacher-forced training loss for [B,T+1,1,64,64] videos."""
        b, t1 = frames.shape[:2]
        z = self.encoder(frames.flatten(0, 1)).clone().view(b, t1, -1)
        one_step = self.step(z[:, :-1], actions).clone()
        state, rollout = z[:, 0], []
        for action in actions.unbind(1):
            state = self.step(state, action).clone()
            rollout.append(state)
        rollout = torch.stack(rollout, 1)
        pred_loss = (one_step - z[:, 1:]).square().mean()
        pred_loss = pred_loss + self.rollout_weight * (rollout - z[:, 1:]).square().mean()
        pooled = self.sigreg(z.flatten(0, 1))
        if self.sigreg_mode == "pooled":
            sigreg = pooled
        elif self.sigreg_mode == "per_time":
            sigreg = torch.stack([self.sigreg(z[:, t]) for t in range(t1)]).mean()
        elif self.sigreg_mode == "both":
            per_time = torch.stack([self.sigreg(z[:, t]) for t in range(t1)]).mean()
            sigreg = 0.5 * (pooled + per_time)
        elif self.sigreg_mode == "pooled_pred":
            predicted = self.sigreg(rollout.flatten(0, 1))
            sigreg = 0.5 * (pooled + predicted)
        else:
            raise ValueError(f"unknown SIGReg mode: {self.sigreg_mode}")
        return pred_loss + self.lamb * sigreg, pred_loss, sigreg, z

    def rollout(self, frame, actions):
        z = self.encoder(frame)
        out = [z]
        for action in actions.unbind(1):
            z = self.step(z, action)
            out.append(z)
        return torch.stack(out, 1)


def save_rollout(path, truth, prediction, actions, fps=4):
    """Write an animated GIF with truth (left) and decoded rollout (right)."""
    frames = []
    truth, prediction = truth.cpu(), prediction.cpu().clamp(0, 1)
    for i, (target, pred) in enumerate(zip(truth, prediction)):
        pair = torch.cat((target, pred), -1).squeeze().clamp(0, 1)
        image = Image.fromarray((pair.numpy() * 255).astype("uint8")).convert("RGB")
        draw = ImageDraw.Draw(image)
        label = "truth | decoded"
        if i:
            action = actions[i - 1].tolist()
            dx, dy = action[:2]
            label += f"   action=({dx:+.2f},{dy:+.2f})"
            if len(action) == 3:
                label += f" rot={action[2]:+.0f}×45°"
        draw.rectangle((0, 0, image.width, 12), fill="black")
        draw.text((2, 1), label, fill="white")
        frames.append(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
    return path


def save_rollout_grid(path, truth, prediction, actions, labels, fps=4):
    """Write a multi-row GIF with one truth/decoded rollout per label."""
    truth, prediction = truth.cpu(), prediction.cpu().clamp(0, 1)
    gif_frames = []
    for time in range(truth.shape[1]):
        canvas = Image.new("RGB", (128, 76 * len(labels)), "black")
        draw = ImageDraw.Draw(canvas)
        for row, label in enumerate(labels):
            pair = torch.cat((truth[row, time], prediction[row, time]), -1).squeeze()
            image = Image.fromarray((pair.clamp(0, 1).numpy() * 255).astype("uint8")).convert(
                "RGB"
            )
            top = row * 76
            canvas.paste(image, (0, top + 12))
            caption = f"digit {label}   truth | decoded"
            if time:
                action = actions[row, time - 1].tolist()
                caption += f"   a=({action[0]:+.1f},{action[1]:+.1f}"
                if len(action) == 3:
                    caption += f",{action[2]:+.0f}x45deg"
                caption += ")"
            draw.text((2, top + 1), caption, fill="white")
        gif_frames.append(canvas)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    return path
