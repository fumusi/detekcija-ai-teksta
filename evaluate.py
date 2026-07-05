"""
Evaluacija modela: metrike, matrica konfuzije, K-fold CV, grafici poređenja.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

from train import train_model, predict, make_loader
from config import CV_FOLDS, RANDOM_STATE, LR, NUM_EPOCHS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'fpr':       fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


def print_and_plot_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str, save_path: str) -> dict:
    metrics = compute_metrics(y_true, y_pred)

    print(f'\n{"=" * 50}')
    print(f'  {model_name}')
    print(f'{"=" * 50}')
    print(f'  Accuracy:                {metrics["accuracy"]:.4f}')
    print(f'  Precision (weighted):    {metrics["precision"]:.4f}')
    print(f'  Recall (weighted):       {metrics["recall"]:.4f}')
    print(f'  F1-score (weighted):     {metrics["f1"]:.4f}')
    print(f'  False Positive Rate:     {metrics["fpr"]:.4f}  (ljudski tekst -> AI)')
    print()
    print(classification_report(y_true, y_pred, target_names=['Covjek (0)', 'AI (1)']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Covjek (0)', 'AI (1)'],
                yticklabels=['Covjek (0)', 'AI (1)'])
    plt.title(f'Matrica konfuzije — {model_name}')
    plt.ylabel('Stvarna klasa')
    plt.xlabel('Prediktovana klasa')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'Sacuvano: {save_path}')

    return metrics


# ─────────────────────────────────────────
# K-fold unakrsna validacija
# ─────────────────────────────────────────

def kfold_knn(X: np.ndarray, y: np.ndarray, k_neighbors: int, n_sample: int = 15_000):
    print(f'\nKNN {CV_FOLDS}-fold CV (podskup {n_sample}):')
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    idx = np.random.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_cv, y_cv = X[idx], y[idx]

    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean', n_jobs=-1)
    cv_acc = cross_val_score(knn, X_cv, y_cv, cv=kf, scoring='accuracy')
    cv_f1  = cross_val_score(knn, X_cv, y_cv, cv=kf, scoring='f1_weighted')

    for i, (a, f) in enumerate(zip(cv_acc, cv_f1), 1):
        print(f'  Fold {i}: Accuracy={a:.4f}, F1={f:.4f}')
    print(f'  Prosjek — Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
    print(f'  Prosjek — F1:       {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}')


def kfold_torch(model_fn, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
                device: torch.device, batch_size: int,
                model_name: str, n_sample: int = 15_000, epochs: int = 5):
    print(f'\n{model_name} {CV_FOLDS}-fold CV (podskup {n_sample}, {epochs} epoha po foldu):')
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n = min(n_sample, len(X_tensor))
    X_cv = X_tensor[:n]
    y_cv = y_tensor[:n]

    accs = []
    for fold, (tr_i, val_i) in enumerate(kf.split(X_cv, y_cv.numpy()), 1):
        X_tr, X_val = X_cv[tr_i], X_cv[val_i]
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        model = model_fn().to(device)
        opt   = optim.Adam(model.parameters(), lr=LR)
        crit  = nn.CrossEntropyLoss()
        loader = make_loader(X_tr, y_tr, batch_size)

        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()

        preds = predict(model, X_val, device)
        acc = accuracy_score(y_val.numpy(), preds)
        accs.append(acc)
        print(f'  Fold {fold}: Accuracy={acc:.4f}')

    print(f'  Prosjek — Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')


# ─────────────────────────────────────────
# Grafici poređenja
# ─────────────────────────────────────────

def plot_comparison(results: dict, save_dir: str = 'plots'):
    df = pd.DataFrame(results).T
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'FPR']

    print('\n' + df.round(4).to_string())

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['steelblue', 'salmon', 'mediumseagreen']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mname, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width, row[metrics].values, width, label=mname, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Vrijednost')
    ax.set_title('Poređenje modela — test skup')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f'{save_dir}/06_poređenje_modela.png'
    plt.savefig(path, dpi=100)
    plt.close()
    print(f'Sacuvano: {path}')

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(df.index, df['FPR'], color=colors)
    for bar, v in zip(bars, df['FPR']):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f'{v:.4f}', ha='center', fontsize=10)
    ax.set_title('False Positive Rate\n(ljudski tekst pogresno oznacen kao AI)')
    ax.set_ylabel('FPR')
    ax.set_ylim(0, max(df['FPR']) * 1.4 + 0.01)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f'{save_dir}/07_fpr.png'
    plt.savefig(path, dpi=100)
    plt.close()
    print(f'Sacuvano: {path}')
