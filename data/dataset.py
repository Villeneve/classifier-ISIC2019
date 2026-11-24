"""Dataset ISIC-2019 binário: um .pt por imagem (uint8, RGB 450x450).

PONTO DE ADAPTAÇÃO: `_scan` descobre (caminho, rótulo, split) a partir da estrutura
de pastas. Layouts suportados (nomes de pasta case-insensitive):
    root/{train,val,test}/{benign,malignant}/*.pt
    root/{benign,malignant}/*.pt            (split estratificado gerado automaticamente)
Pastas de classe aceitas: benign/benigno/0  e  malignant/maligno/1.
"""
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

MAL = {"1", "malignant", "maligno", "malign", "mal"}
BEN = {"0", "benign", "benigno", "ben"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def _scan(root):
    """Retorna {split: [(path, label), ...]}; split 'all' se não houver pastas de split."""
    out = {}
    for dp, _, files in os.walk(root):
        rel = os.path.relpath(dp, root).lower().split(os.sep)
        label, split = None, "all"
        for part in rel:
            if part in MAL:
                label = 1
            elif part in BEN:
                label = 0
            elif part in SPLIT_ALIASES:
                split = SPLIT_ALIASES[part]
        for f in files:
            if not f.endswith(".pt"):
                continue
            if label is None:
                raise ValueError(f"Não consegui inferir o rótulo de {os.path.join(dp, f)}; ajuste _scan().")
            out.setdefault(split, []).append((os.path.join(dp, f), label))
    for v in out.values():
        v.sort()
    if not out:
        raise FileNotFoundError(f"Nenhum .pt encontrado em {root}")
    return out


def _stratified_split(items, fracs, seed):
    """Divide `items` em len(fracs) partes estratificadas por rótulo (última recebe o resto)."""
    rng = random.Random(seed)
    parts = [[] for _ in fracs]
    for lab in (0, 1):
        grp = [it for it in items if it[1] == lab]
        rng.shuffle(grp)
        start = 0
        for i, fr in enumerate(fracs):
            end = len(grp) if i == len(fracs) - 1 else start + int(round(fr * len(grp)))
            parts[i] += grp[start:end]
            start = end
    for p in parts:
        rng.shuffle(p)
    return parts


def make_splits(cfg):
    found = _scan(cfg.data_root)
    if {"train", "val", "test"} <= found.keys():
        splits = {k: found[k] for k in ("train", "val", "test")}
    elif "all" in found:
        tr, va, te = _stratified_split(
            found["all"], [1 - cfg.val_frac - cfg.test_frac, cfg.val_frac, cfg.test_frac], cfg.seed)
        splits = {"train": tr, "val": va, "test": te}
    elif {"train", "test"} <= found.keys():
        tr, va = _stratified_split(found["train"], [1 - cfg.val_frac, cfg.val_frac], cfg.seed)
        splits = {"train": tr, "val": va, "test": found["test"]}
    else:
        raise ValueError(f"Layout de splits não reconhecido: {list(found)}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "split.json"), "w") as f:
        json.dump(splits, f)

    if cfg.subset > 0:
        rng = random.Random(cfg.seed)
        splits = {k: rng.sample(v, min(cfg.subset, len(v))) for k, v in splits.items()}
    return splits


class PTDataset(Dataset):
    """Retorna (uint8 [3,H,W], rótulo float). Sem augmentation: ela roda na GPU."""

    def __init__(self, items, cache_ram=False):
        self.items = items
        self.cache = [self._load(p) for p, _ in items] if cache_ram else None

    @staticmethod
    def _load(path):
        x = torch.load(path, map_location="cpu")
        if isinstance(x, dict):
            x = next(v for v in x.values() if torch.is_tensor(v))
        if x.ndim == 3 and x.shape[-1] == 3 and x.shape[0] != 3:  # HWC -> CHW
            x = x.permute(2, 0, 1)
        if x.dtype != torch.uint8:
            x = (x * 255 if x.is_floating_point() and x.max() <= 1.0 else x).clamp(0, 255).to(torch.uint8)
        return x.contiguous()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.cache[i] if self.cache is not None else self._load(self.items[i][0])
        return x, torch.tensor(float(self.items[i][1]))


def make_loaders(cfg, batch_size):
    splits = make_splits(cfg)
    loaders = {}
    for name, items in splits.items():
        ds = PTDataset(items, cfg.cache_ram)
        train = name == "train"
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=train, drop_last=train and len(ds) > batch_size,
            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.num_workers > 0)
    n_pos = sum(l for _, l in splits["train"])
    n_neg = len(splits["train"]) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    counts = {k: (len(v), sum(l for _, l in v)) for k, v in splits.items()}
    return loaders, pos_weight, counts
