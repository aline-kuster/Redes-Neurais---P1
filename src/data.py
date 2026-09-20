"""
data_utils.py
--------------
Carregamento do dataset e separação em treino / validação / teste.

treino ajusta os pesos da rede;
validação serve para escolher os hiperparâmetros;
teste é usado uma única vez no final pra reportar o desempenho.
com a proporção 10/10/80 fica mais fácil enxergar
overfitting, que é o que os estudos de ablação tentam corrigir depois.
"""

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """Fixa a semente do numpy e do torch toda execução começa exatamente do mesmo jeito"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(csv_path: str):
    """Lê o CSV (colunas 'x' e 'y') e devolve dois arrays numpy 1D."""
    df = pd.read_csv(csv_path)
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    return x, y


def split_dataset(x, y, train_frac=0.1, val_frac=0.1, seed=42):
    """
    Embaralha os índices e separa em treino/validação/teste segundo as
    frações pedidas. O resto (1 - train_frac - val_frac) vai para teste.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)  # embaralha os índices 0..n-1

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    splits = {
        "train": (x[train_idx], y[train_idx]),
        "val": (x[val_idx], y[val_idx]),
        "test": (x[test_idx], y[test_idx]),
    }
    return splits


def standardize(x_train, *other_arrays):
    """
    Calcula média/desvio padrão usando SOMENTE x_train, e aplica essa mesma
    transformação em x_train e em qualquer outro array passado (val, test).
    """
    mean = x_train.mean()
    std = x_train.std()
    std = std if std > 1e-8 else 1.0  # proteção contra divisão por zero

    def _apply(a):
        return (a - mean) / std

    transformed = [_apply(x_train)] + [_apply(a) for a in other_arrays]
    return (*transformed, mean, std)


def to_tensor(x, y):
    """Converte arrays numpy 1D em tensores torch de shape (N, 1),
    formato que a camada Linear do PyTorch espera (batch, features)."""
    x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return x_t, y_t


def prepare_data(csv_path: str, train_frac=0.1, val_frac=0.1, seed=42):
    """
    Faz tudo: carrega, separa treino/val/teste, padroniza o x (com
    estatísticas do treino) e devolve tensores prontos pra treinar.
    """
    set_seed(seed)
    x, y = load_dataset(csv_path)
    splits = split_dataset(x, y, train_frac, val_frac, seed) '''embaralha e separa esses 300 pontos em 3 grupos: 30 pra treino, 30 pra validação, 240 pra teste'''

    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    x_train_std, x_val_std, x_test_std, mean, std = standardize(x_train, x_val, x_test)

    data = {
        "train": to_tensor(x_train_std, y_train),
        "val": to_tensor(x_val_std, y_val),
        "test": to_tensor(x_test_std, y_test),
        "x_mean": mean,
        "x_std": std,
    }
    return data

