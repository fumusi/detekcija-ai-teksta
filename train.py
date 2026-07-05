"""
Treniranje i predikcija PyTorch modela.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def train_model(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                num_epochs: int, device: torch.device) -> list:
    """Trening petlja. Vraca listu loss vrijednosti po epohama."""
    history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        history.append(avg)
        print(f'  Epoha [{epoch+1}/{num_epochs}], Loss: {avg:.4f}')
    return history


def predict(model: nn.Module, X_tensor: torch.Tensor,
            device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Batch predikcija — vraca niz klasa."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            out = model(X_tensor[i:i+batch_size].to(device))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
    return np.array(preds)


def make_loader(X_tensor: torch.Tensor, y_tensor: torch.Tensor,
                batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(X_tensor, y_tensor),
                      batch_size=batch_size, shuffle=shuffle)


def plot_loss(history: list, title: str, save_path: str, color: str = 'steelblue'):
    plt.figure(figsize=(6, 4))
    plt.plot(history, color=color)
    plt.title(title)
    plt.xlabel('Epoha')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'Sacuvano: {save_path}')
