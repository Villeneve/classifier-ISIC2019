"""Dataset ISIC-2019 binário a partir dos arquivos originais (JPEG + CSV).

Espera em `data_root` (link simbólico `ISIC-2019/`):
    train/ISIC_2019_Training_Input/*.jpg   train/ISIC_2019_Training_GroundTruth.csv   train/ISIC_2019_Training_Metadata.csv
    test/ISIC_2019_Test_Input/*.jpg        test/ISIC_2019_Test_GroundTruth.csv

Pipeline:
1. Os JPEGs (tamanhos variados) são decodificados uma única vez (nvjpeg na GPU, por causa da CPU fraca),
   redimensionados (lado menor -> `cache_size`, recorte central quadrado) e gravados em um cache
   uint8 [N,3,S,S] (.npy, lido com mmap) em `cache_dir`. As execuções seguintes só leem o cache.
2. Rótulo binário a partir das 8 classes: maligno = MEL, BCC, SCC; benigno = NV, BKL, DF, VASC.
   AK (queratose actínica, pré-maligna) é controlada por `cfg.ak` (benign | malignant | drop).
3. Validação = parte do treino, separada por *lesão* (lesion_id do Metadata) para evitar vazamento
   entre imagens da mesma lesão, estratificada por classe. O teste é o conjunto `test/`.
"""
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image, decode_jpeg
from tqdm import tqdm

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MALIGNANT = {"MEL", "BCC", "SCC"}

_FILES = {
    "train": ("train", "ISIC_2019_Training_Input", "ISIC_2019_Training_GroundTruth.csv",
              "ISIC_2019_Training_Metadata.csv"),
    "test": ("test", "ISIC_2019_Test_Input", "ISIC_2019_Test_GroundTruth.csv", None),
}


# ----------------------------------------------------------------------------- CSVs
def _read_gt(csv_path):
    """Retorna (ids, índices de classe 0..7); descarta linhas UNK ou sem classe."""
    ids, cls = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            vals = [float(row[c]) for c in CLASSES]
            if sum(vals) == 0 or float(row.get("UNK", 0) or 0) > 0:
                continue
            ids.append(row["image"])
            cls.append(int(np.argmax(vals)))
    return ids, np.array(cls)


def _read_groups(meta_path, ids):
    """Grupo de cada imagem = lesion_id (se houver), senão o próprio id da imagem."""
    lesion = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("lesion_id"):
                    lesion[row["image"]] = row["lesion_id"]
    return [lesion.get(i, i) for i in ids]


def _binary_labels(cls, ak):
    """cls (0..7) -> (rótulos 0/1, máscara das amostras mantidas)."""
    mal = set(MALIGNANT) | ({"AK"} if ak == "malignant" else set())
    lut = np.array([1 if c in mal else 0 for c in CLASSES])
    keep = np.ones(len(cls), dtype=bool)
    if ak == "drop":
        keep = cls != CLASSES.index("AK")
    return lut[cls], keep


# ----------------------------------------------------------------------------- cache
def _decode_square(path, size, device):
    """JPEG -> uint8 [3,size,size]: lado menor = size (antialias) + recorte central."""
    with open(path, "rb") as f:
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
    try:
        img = decode_jpeg(data, device=device, mode=ImageReadMode.RGB)
    except Exception:  # noqa: BLE001  (sem nvjpeg, CMYK etc.): decodifica na CPU
        img = decode_image(data, mode=ImageReadMode.RGB).to(device)
    h, w = img.shape[-2:]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    x = F.interpolate(img[None].float(), size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    top, left = (nh - size) // 2, (nw - size) // 2
    return x[0, :, top:top + size, left:left + size].round().clamp(0, 255).to(torch.uint8).cpu()


def _build_cache(img_dir, ids, size, device, cache_path):
    missing = [i for i in ids if not os.path.exists(os.path.join(img_dir, i + ".jpg"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imagens ausentes em {img_dir} (ex.: {missing[0]}.jpg)")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(ids), 3, size, size))
    with torch.no_grad():
        for k, i in enumerate(tqdm(ids, desc=f"cache {os.path.basename(cache_path)}")):
            mm[k] = _decode_square(os.path.join(img_dir, i + ".jpg"), size, device).numpy()
    mm.flush()
    del mm
    os.replace(tmp, cache_path)


def load_split(cfg, split):
    """Retorna dict(images=memmap [N,3,S,S], labels [N], keep [índices], ids, groups)."""
    sub, img_name, gt_name, meta_name = _FILES[split]
    root = os.path.join(cfg.data_root, sub)
    ids, cls = _read_gt(os.path.join(root, gt_name))
    groups = _read_groups(os.path.join(root, meta_name) if meta_name else None, ids)

    tag = ""
    if cfg.subset > 0 and cfg.subset < len(ids):  # smoke test: cache próprio, só com N imagens
        sel = sorted(random.Random(cfg.seed).sample(range(len(ids)), cfg.subset))
        ids, cls, groups = [ids[i] for i in sel], cls[sel], [groups[i] for i in sel]
        tag = f"_sub{cfg.subset}"
    cache = os.path.join(cfg.cache_dir, f"{split}_{cfg.cache_size}{tag}.npy")
    ids_file = cache.replace(".npy", ".json")

    valid = (os.path.exists(cache) and os.path.exists(ids_file) and not cfg.rebuild_cache
             and json.load(open(ids_file)) == ids)
    if not valid:
        if os.path.exists(ids_file):
            os.remove(ids_file)
        _build_cache(os.path.join(root, img_name), ids, cfg.cache_size, cfg.device, cache)
        with open(ids_file, "w") as f:
            json.dump(ids, f)  # gravado por último: marca o cache como completo

    images = np.load(cache, mmap_mode=None if cfg.cache_ram else "r")
    labels, keep = _binary_labels(cls, cfg.ak)
    return {"images": images, "labels": labels, "keep": np.flatnonzero(keep).tolist(),
            "ids": ids, "groups": groups}


# ----------------------------------------------------------------------------- splits
def _group_split(idx, labels, groups, val_frac, seed):
    """Divide `idx` em (treino, validação) sem separar imagens do mesmo grupo (lesão),
    estratificando pelo rótulo do grupo (maligno se qualquer imagem for maligna)."""
    rng = random.Random(seed)
    by_group = {}
    for i in idx:
        by_group.setdefault(groups[i], []).append(i)
    tr, va = [], []
    for lab in (0, 1):
        gs = sorted(g for g, m in by_group.items() if int(max(labels[i] for i in m)) == lab)
        rng.shuffle(gs)
        n_val = int(round(val_frac * len(gs)))
        for g in gs[:n_val]:
            va += by_group[g]
        for g in gs[n_val:]:
            tr += by_group[g]
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


class ArrayDataset(Dataset):
    """Retorna (uint8 [3,S,S], rótulo float). Sem augmentation: ela roda na GPU."""

    def __init__(self, images, labels, indices):
        self.images, self.labels, self.indices = images, labels, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return torch.from_numpy(np.array(self.images[j])), torch.tensor(float(self.labels[j]))


def make_loaders(cfg, batch_size):
    tr = load_split(cfg, "train")
    te = load_split(cfg, "test")

    tr_idx, va_idx = _group_split(tr["keep"], tr["labels"], tr["groups"], cfg.val_frac, cfg.seed)
    te_idx = te["keep"]
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "split.json"), "w") as f:
        json.dump({"train": [tr["ids"][i] for i in tr_idx], "val": [tr["ids"][i] for i in va_idx],
                   "test": [te["ids"][i] for i in te_idx]}, f)

    parts = {"train": (tr, tr_idx), "val": (tr, va_idx), "test": (te, te_idx)}
    loaders, counts = {}, {}
    for name, (d, idx) in parts.items():
        train = name == "train"
        loaders[name] = DataLoader(
            ArrayDataset(d["images"], d["labels"], idx), batch_size=batch_size, shuffle=train,
            drop_last=train and len(idx) > batch_size, num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=cfg.num_workers > 0)
        counts[name] = (len(idx), int(d["labels"][idx].sum()))

    n, n_pos = counts["train"]
    pos_weight = (n - n_pos) / max(n_pos, 1)
    return loaders, pos_weight, counts
