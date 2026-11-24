"""Dataset ISIC-2019 binário a partir de tensores .pth.

Espera em `data_root` (ex.: o link simbólico `pth/`):
    train_images.pth  [N,3,450,450] uint8      train_labels.pth  [N] ou [N,1]  (0 benigno, 1 maligno)
    test_images.pth   [M,3,450,450] uint8      test_labels.pth   [M] ou [M,1]
Não há split de validação pronto: ele é separado do treino (estratificado, seed fixa).
Os arquivos são abertos com mmap, então não ocupam a RAM inteira.
"""
import json
import os
import pickle
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors


def _load_tensor(path, mmap):
    """Carrega o .pth como torch.Tensor comum (os arquivos podem ser tv_tensors.Image)."""
    kw = dict(map_location="cpu", mmap=mmap)
    try:
        # Libera só a classe tv_tensors.Image, mantendo o carregamento seguro (weights_only).
        with torch.serialization.safe_globals([tv_tensors.Image, tv_tensors.Mask]):
            x = torch.load(path, **kw)
    except pickle.UnpicklingError as e:
        print(f"[aviso] {os.path.basename(path)}: carregamento seguro falhou ({str(e)[:120]}...); "
              "usando weights_only=False (arquivo próprio, confiável).")
        x = torch.load(path, weights_only=False, **kw)
    if isinstance(x, dict):
        x = next(v for v in x.values() if torch.is_tensor(v))
    return x.as_subclass(torch.Tensor)  # tira a subclasse (tv_tensors) e mantém o mmap


def load_split_tensors(root, split, cache_ram=False):
    """Retorna (imagens [N,3,H,W], rótulos int64 [N])."""
    imgs = _load_tensor(os.path.join(root, f"{split}_images.pth"), mmap=not cache_ram)
    labels = _load_tensor(os.path.join(root, f"{split}_labels.pth"), mmap=False).reshape(-1).long()
    if len(imgs) != len(labels):
        raise ValueError(f"{split}: {len(imgs)} imagens vs {len(labels)} rótulos")
    if not set(labels.unique().tolist()) <= {0, 1}:
        raise ValueError(f"{split}: rótulos não são binários: {labels.unique().tolist()}")
    return imgs, labels


def _stratified_split(labels, val_frac, seed):
    """Separa índices em (treino, validação) mantendo a proporção de classes."""
    rng = random.Random(seed)
    tr, va = [], []
    for lab in (0, 1):
        idx = [i for i, l in enumerate(labels.tolist()) if l == lab]
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        va += idx[:n_val]
        tr += idx[n_val:]
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


class TensorDataset(Dataset):
    """Retorna (uint8 [3,H,W], rótulo float). Sem augmentation: ela roda na GPU."""

    def __init__(self, images, labels, indices):
        self.images, self.labels, self.indices = images, labels, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = self.images[j]
        if x.dtype != torch.uint8:
            x = (x * 255 if x.is_floating_point() and x.max() <= 1.0 else x).clamp(0, 255).to(torch.uint8)
        return x.clone(), self.labels[j].float()  # clone: tira o tensor do mmap


def make_loaders(cfg, batch_size):
    tr_imgs, tr_labels = load_split_tensors(cfg.data_root, "train", cfg.cache_ram)
    te_imgs, te_labels = load_split_tensors(cfg.data_root, "test", cfg.cache_ram)

    tr_idx, va_idx = _stratified_split(tr_labels, cfg.val_frac, cfg.seed)
    te_idx = list(range(len(te_labels)))
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "split.json"), "w") as f:
        json.dump({"train": tr_idx, "val": va_idx, "test_size": len(te_idx)}, f)

    if cfg.subset > 0:  # smoke test
        rng = random.Random(cfg.seed)
        tr_idx, va_idx, te_idx = (rng.sample(s, min(cfg.subset, len(s))) for s in (tr_idx, va_idx, te_idx))

    parts = {"train": (tr_imgs, tr_labels, tr_idx),
             "val": (tr_imgs, tr_labels, va_idx),
             "test": (te_imgs, te_labels, te_idx)}
    loaders, counts = {}, {}
    for name, (imgs, labels, idx) in parts.items():
        train = name == "train"
        loaders[name] = DataLoader(
            TensorDataset(imgs, labels, idx), batch_size=batch_size, shuffle=train,
            drop_last=train and len(idx) > batch_size, num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=cfg.num_workers > 0)
        counts[name] = (len(idx), int(labels[idx].sum()))

    n, n_pos = counts["train"]
    pos_weight = (n - n_pos) / max(n_pos, 1)
    return loaders, pos_weight, counts
