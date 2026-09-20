"""Métricas para regressão."""

import torch


def regression_metrics(model, x, y):
    """Calcula MSE, RMSE, MAE e R² para um modelo de regressão."""
    model.eval()
    with torch.no_grad():
        pred = model(x)
        if pred.shape != y.shape:
            pred = pred.reshape_as(y)
        diff = pred - y # diferença entre o previsto e o real, ponto a ponto
        mse = torch.mean(diff ** 2).item()  # erro médio ao quadrado
        rmse = torch.sqrt(torch.mean(diff ** 2)).item()  # raiz do MSE
        mae = torch.mean(torch.abs(diff)).item()  # erro médio absoluto (sem elevar ao quadrado)

        y_mean = torch.mean(y)
        ss_res = torch.sum(diff ** 2).item()
        ss_tot = torch.sum((y - y_mean) ** 2).item()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def format_metrics(metrics):
    """Formata um dicionário de métricas para impressão."""
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name.upper():>8}: {value:.6f}")
    return "\n".join(lines)
