"""Loop de treino/validação com bf16, early stopping e checkpoint do melhor modelo."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.augment import augment_train, preprocess_eval
from engine.metrics import roc_auc, compute_metrics
from models.factory import set_backbone_trainable


def _prep(x, y, device, size, train):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    x = augment_train(x, size) if train else preprocess_eval(x, size)
    return x.contiguous(memory_format=torch.channels_last), y


@torch.no_grad()
def predict(fwd, loader, device, size, criterion=None, desc="eval"):
    """Retorna (probabilidades, rótulos, loss média) como numpy."""
    fwd.eval()
    probs, labels, total, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = _prep(x, y, device, size, train=False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = fwd(x).squeeze(1)
        logits = logits.float()
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
            n += len(y)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    return (torch.cat(probs).numpy(), torch.cat(labels).numpy(), total / max(n, 1))


def fit(model, fwd, loaders, cfg, size, pos_weight, out_dir):
    """Treina `model` (fwd = model ou versão compilada). Salva best.pt e history.json."""
    device = cfg.device
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    hist = {k: [] for k in ("train_loss", "val_loss", "val_auc", "val_bacc", "lr")}
    best_auc, bad, ckpt = -1.0, 0, os.path.join(out_dir, "best.pt")

    for ep in range(cfg.epochs):
        set_backbone_trainable(model, ep >= cfg.freeze_epochs)
        model.train()
        run, n = 0.0, 0
        bar = tqdm(loaders["train"], desc=f"ep {ep + 1}/{cfg.epochs}", leave=False)
        for x, y in bar:
            x, y = _prep(x, y, device, size, train=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = fwd(x).squeeze(1)
            loss = criterion(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
            n += len(y)
            bar.set_postfix(loss=f"{run / n:.4f}")
        sched.step()

        p, y, vloss = predict(fwd, loaders["val"], device, size, criterion, desc="val")
        auc = roc_auc(y, p)
        bacc = compute_metrics(y, p)["balanced_accuracy"]
        for k, v in zip(hist, (run / max(n, 1), vloss, auc, bacc, opt.param_groups[0]["lr"])):
            hist[k].append(float(v))
        print(f"ep {ep + 1:02d} | train {hist['train_loss'][-1]:.4f} | val {vloss:.4f} "
              f"| AUC {auc:.4f} | bAcc {bacc:.4f}")

        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping na época {ep + 1}")
                break

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return hist
