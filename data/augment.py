"""Pré-processamento e augmentation em batch na GPU (CPU do alvo é fraca)."""
import math

import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _norm(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _resize(x, size):
    return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)


@torch.no_grad()
def preprocess_eval(x, size):
    """x: uint8 [B,3,H,W] na GPU -> float normalizado [B,3,size,size]."""
    return _norm(_resize(x.float() / 255.0, size))


@torch.no_grad()
def augment_train(x, size):
    """Augmentation por amostra: rotação livre, flip, zoom/translação, brilho/contraste/saturação."""
    B, dev = x.shape[0], x.device
    x = _resize(x.float() / 255.0, int(size * 1.15))  # reduz antes de amostrar (evita aliasing)

    u = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    ang = u(-math.pi, math.pi)
    scale = u(0.7, 1.0)                                  # fração do campo de visão mantida
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    tx, ty = u(-0.1, 0.1), u(-0.1, 0.1)
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], 1),
        torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    v = lambda: u(0.8, 1.2).view(B, 1, 1, 1)
    x = x * v()                                           # brilho
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * v() + mean                           # contraste
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * v() + gray                           # saturação
    return _norm(x.clamp(0, 1))
