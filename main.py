"""Compara arquiteturas na classificação binária (benigno x maligno) do ISIC-2019.

Exemplos:
    python main.py --data-root /caminho/dataset --models resnet50 --epochs 1 --subset 200   # smoke test
    python main.py --data-root /caminho/dataset                                              # todas
"""
import argparse
import gc
import json
import os

import torch
from torchinfo import summary

from config import ALL_MODELS, Config
from data.dataset import make_loaders
from engine.metrics import compute_metrics, roc_auc, roc_curve, youden_threshold
from engine.train import fit, predict
from models.factory import build_model
from utils.plots import plot_comparison, plot_confusion, plot_history, plot_roc
from utils.seed import set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=Config.data_root)
    ap.add_argument("--out-dir", default=Config.out_dir)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--patience", type=int, default=Config.patience)
    ap.add_argument("--freeze-epochs", type=int, default=Config.freeze_epochs)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=Config.num_workers)
    ap.add_argument("--subset", type=int, default=0, help="N amostras por split (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--cache-ram", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=Config.seed)
    a = ap.parse_args()
    return Config(data_root=a.data_root, out_dir=a.out_dir, models=a.models, pretrained=not a.no_pretrained,
                  epochs=a.epochs, lr=a.lr, patience=a.patience, freeze_epochs=a.freeze_epochs,
                  num_workers=a.num_workers, cache_ram=a.cache_ram, subset=a.subset,
                  batch_size=a.batch_size, img_size=a.img_size, compile=a.compile, seed=a.seed)


def run_model(name, cfg):
    out = os.path.join(cfg.out_dir, name)
    os.makedirs(out, exist_ok=True)
    device, size = cfg.device, cfg.size_for(name)
    set_seed(cfg.seed)

    loaders, pos_weight, counts = make_loaders(cfg, cfg.batch_for(name))
    print(f"\n=== {name} | size {size} | batch {cfg.batch_for(name)} | pos_weight {pos_weight:.2f}")
    print("splits (n, n_maligno):", counts)

    model = build_model(name, cfg.pretrained).to(device)
    info = summary(model, input_size=(1, 3, size, size), device=device, verbose=0)
    print(info)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(str(info))

    model = model.to(memory_format=torch.channels_last)
    fwd = torch.compile(model) if cfg.compile else model

    hist = fit(model, fwd, loaders, cfg, size, pos_weight, out)
    plot_history(hist, os.path.join(out, "history.png"))

    # Limiar escolhido na validação (Youden) e aplicado no teste.
    pv, yv, _ = predict(fwd, loaders["val"], device, size, desc="val")
    thr = youden_threshold(yv, pv)
    pt, yt, _ = predict(fwd, loaders["test"], device, size, desc="test")
    res = {"test_youden": compute_metrics(yt, pt, thr), "test_0.5": compute_metrics(yt, pt, 0.5),
           "params": info.total_params}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    fpr, tpr, _ = roc_curve(yt, pt)
    plot_roc(fpr, tpr, roc_auc(yt, pt), os.path.join(out, "roc.png"), f"ROC - {name}")
    plot_confusion(res["test_youden"]["confusion"], os.path.join(out, "confusion.png"),
                   f"{name} (limiar {thr:.2f})")

    del model, fwd, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    cfg = parse_args()
    torch.backends.cudnn.benchmark = True
    print(f"device: {cfg.device} | torch {torch.__version__}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    summary_all = {}
    for name in cfg.models:
        summary_all[name] = run_model(name, cfg)
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
            json.dump(summary_all, f, indent=2)

    table = {n: r["test_youden"] for n, r in summary_all.items()}
    plot_comparison(table, os.path.join(cfg.out_dir, "comparison.png"))
    print(f"\n{'modelo':22s} {'AUC':>6s} {'bAcc':>6s} {'Sens':>6s} {'Spec':>6s} {'F1':>6s}")
    for n, m in sorted(table.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"{n:22s} {m['auc']:6.3f} {m['balanced_accuracy']:6.3f} "
              f"{m['sensitivity']:6.3f} {m['specificity']:6.3f} {m['f1']:6.3f}")


if __name__ == "__main__":
    main()
