"""
plots.py
--------
Todas as funções de gráfico do projeto.
"""

import os
import matplotlib.pyplot as plt


def plot_train_val(history, title, save_path):
    """Curva de treino x validação, uma única configuração (estilo do baseline)."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="treino (MSE)", color="#2563eb")
    plt.plot(history["val_loss"], label="validação (MSE)", color="#dc2626")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_before_after(baseline_history, other_history, other_label, title, save_path):
    """Compara baseline (tracejado) com uma variante (sólido) — treino e validação das duas."""
    plt.figure(figsize=(8, 5))
    plt.plot(baseline_history["train_loss"], label="treino — baseline", color="#93c5fd", linestyle="--")
    plt.plot(baseline_history["val_loss"], label="validação — baseline", color="#fca5a5", linestyle="--")
    plt.plot(other_history["train_loss"], label=f"treino — {other_label}", color="#2563eb")
    plt.plot(other_history["val_loss"], label=f"validação — {other_label}", color="#dc2626")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_comparison(names, r2_values, mse_values, save_path, colors=None):
    """Gráfico de barras final: R² e MSE de todas as configurações lado a lado."""
    if colors is None:
        colors = ["#6b7280", "#2563eb", "#7c3aed", "#16a34a", "#ea580c", "#0891b2"][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(names, r2_values, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("R² (teste)")
    axes[0].set_title("R² por configuração")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + (0.005 if v >= 0 else -0.012), f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(names, mse_values, color=colors)
    axes[1].set_ylabel("MSE (teste)")
    axes[1].set_title("MSE por configuração")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(mse_values):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Comparação final: baseline x estudos de ablação")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_val_comparison(histories, title, save_path):
    """Figura 2 do template: sobrepõe a curva de validação de várias
    configurações num único gráfico (baseline x L1 x L2 x dropout x
    momentum x combinado)."""
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(history["val_loss"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("MSE (validação)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_parity(y_true, pred_baseline, pred_best, best_label, title, save_path):
    """Figura 3 do template: dispersão y real x y previsto (linha y=x),
    comparando o baseline com o melhor modelo, no conjunto de teste."""
    y_true_np = y_true.detach().numpy().flatten()
    pred_baseline_np = pred_baseline.detach().numpy().flatten()
    pred_best_np = pred_best.detach().numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_np, pred_baseline_np, alpha=0.4, label="baseline", color="#93c5fd")
    plt.scatter(y_true_np, pred_best_np, alpha=0.4, label=f"melhor modelo ({best_label})", color="#dc2626")

    lims = [
        min(y_true_np.min(), pred_baseline_np.min(), pred_best_np.min()),
        max(y_true_np.max(), pred_baseline_np.max(), pred_best_np.max()),
    ]
    plt.plot(lims, lims, "k--", linewidth=1, label="y = x (previsão perfeita)")

    plt.xlabel("y real")
    plt.ylabel("y previsto")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_residuals(y_true, y_pred, label, title, save_path):
    """Figura 4 do template: resíduos (y real - y previsto) do melhor
    modelo, contra o valor previsto."""
    y_true_np = y_true.detach().numpy().flatten()
    y_pred_np = y_pred.detach().numpy().flatten()
    residuals = y_true_np - y_pred_np

    plt.figure(figsize=(7, 4.5))
    plt.scatter(y_pred_np, residuals, alpha=0.5, color="#7c3aed")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("y previsto")
    plt.ylabel("resíduo (y real − y previsto)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()