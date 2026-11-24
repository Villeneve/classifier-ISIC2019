import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], label="treino")
    ax[0].plot(ep, hist["val_loss"], label="validação")
    ax[0].set(title="Loss", xlabel="época")
    ax[1].plot(ep, hist["val_auc"], label="AUC")
    ax[1].plot(ep, hist["val_bacc"], label="Acc. balanceada")
    ax[1].set(title="Validação", xlabel="época")
    ax[2].plot(ep, hist["lr"])
    ax[2].set(title="Learning rate", xlabel="época")
    for a in ax[:2]:
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(fpr, tpr, auc, path, title="ROC"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title=title, xlabel="1 - especificidade", ylabel="sensibilidade")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, path, title="Matriz de confusão"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(title=title, xlabel="previsto", ylabel="real",
           xticks=[0, 1], yticks=[0, 1],
           xticklabels=["benigno", "maligno"], yticklabels=["benigno", "maligno"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(results, path, keys=("auc", "balanced_accuracy", "sensitivity", "specificity", "f1")):
    """results: {modelo: {métrica: valor}} (teste)."""
    names = list(results)
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(names)) + i * w, [results[n][k] for n in names], w, label=k)
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(ylim=(0, 1), title="Comparação no conjunto de teste")
    ax.legend(ncol=len(keys), fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
