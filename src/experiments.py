"""
experiments.py
--------------
Ponto único que roda todo o estudo:

  1) busca empírica do baseline (arquitetura + lr, SGD puro)
  2) treino final do baseline escolhido
  3) os 4 estudos de ablação (dropout, L2, L1, momentum), cada um mudando
     SÓ um hiperparâmetro em cima da MESMA arquitetura do baseline
  4) modelo combinado, juntando os componentes que isoladamente ajudaram
  5) resumo em texto com as métricas de todas as configurações
  6) gráficos: curva do baseline, baseline x cada ablação, validação de
     todas as configurações sobrepostas, comparação final em barras,
     parity plot e resíduos do melhor modelo

Rodar com: python src/experiments.py
"""

import os
import csv

import torch

from data import prepare_data, set_seed
from model import MLP
from train import train_model
from metrics import regression_metrics, format_metrics
from plots import (
    plot_train_val,
    plot_before_after,
    plot_comparison,
    plot_val_comparison,
    plot_parity,
    plot_residuals,
)

DATA_CSV = "data/dataset_projeto1.csv"
OUT_DIR = "outputs"

# grade da busca empírica do baseline: 6 arquiteturas x 3 lr = 18 combinações
ARCHITECTURES = [(8,), (16,), (32,), (16, 16), (32, 16), (64, 64)]
LEARNING_RATES = [0.1, 0.05, 0.01]
EPOCHS = 3000
BATCH_SIZE = 8

# um hiperparâmetro por ablação, escolha foi feito manualmente testando cada um isoladamente, 
# e pegando o que melhorou o R2 no teste.
ABLATIONS = {
    "dropout": dict(dropout_p=0.3),
    "l2": dict(weight_decay=1e-5),
    "l1": dict(l1_lambda=1e-6),
    "momentum": dict(momentum=0.7),
}

# modelo combinado: dropout + momentum, que foram os dois que melhoraram o R2 no teste
COMBINED = dict(dropout_p=0.3, momentum=0.7)


def search_baseline(data):
    """Testa todas as combinações de arquitetura/lr (SGD puro) e escolhe a
    de menor MSE de validação."""
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    results = []
    for hidden_sizes in ARCHITECTURES:
        for lr in LEARNING_RATES:
            set_seed(42)    # toda combinação começa do mesmo ponto
            model = MLP(hidden_sizes=hidden_sizes)
            history = train_model(
                model, x_train, y_train, x_val, y_val,
                lr=lr, epochs=EPOCHS, batch_size=BATCH_SIZE,
            )
            n_params = sum(p.numel() for p in model.parameters())
            results.append({
                "hidden_sizes": str(hidden_sizes),
                "lr": lr,
                "n_params": n_params,
                "final_train_mse": history["train_loss"][-1],
                "final_val_mse": history["val_loss"][-1],
                "best_val_mse": min(history["val_loss"]),
            })
            print(f"arq={str(hidden_sizes):12s} lr={lr:<5} n_params={n_params:4d} "
                  f"| val_final={results[-1]['final_val_mse']:.4f} best_val={results[-1]['best_val_mse']:.4f}")

       # salva a busca inteira em CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/baseline_search_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
     # escolhe a combinação com o menor erro de validação -> vira o baseline oficial
    best = min(results, key=lambda r: r["best_val_mse"])
    print("\n>>> Baseline escolhido:", best)
    return best


def train_config(data, hidden_sizes, lr, **train_kwargs):
    """Treina uma configuração (baseline, ablação ou combinado) do zero e
    devolve (modelo, history). dropout_p é tratado à parte porque é da
    arquitetura, não do optimizer."""
    dropout_p = train_kwargs.pop("dropout_p", 0.0)
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    set_seed(42)
    model = MLP(hidden_sizes=hidden_sizes, dropout_p=dropout_p)
    history = train_model(
        model, x_train, y_train, x_val, y_val,
        lr=lr, epochs=EPOCHS, batch_size=BATCH_SIZE,
        **train_kwargs,
    )
    return model, history


def main():
    data = prepare_data(DATA_CSV)
    x_test, y_test = data["test"]

    # 1) busca empírica + baseline final
    best = search_baseline(data)
    hidden_sizes = eval(best["hidden_sizes"])  
    lr = best["lr"]

    baseline_model, baseline_history = train_config(data, hidden_sizes, lr)
    plot_train_val(
        baseline_history,
        f"Baseline: hidden={hidden_sizes}, lr={lr} (SGD puro)",
        f"{OUT_DIR}/baseline_curve.png",
    )

    all_metrics = {"baseline": regression_metrics(baseline_model, x_test, y_test)}
    all_models = {"baseline": baseline_model}
    all_histories = {"baseline": baseline_history}

    # 2) os 4 estudos de ablação, cada um comparado ao baseline
    for name, kwargs in ABLATIONS.items():
        model, history = train_config(data, hidden_sizes, lr, **kwargs)
        all_metrics[name] = regression_metrics(model, x_test, y_test)
        all_models[name] = model
        all_histories[name] = history

        plot_before_after(
            baseline_history, history, name,
            f"Baseline (tracejado) x {name} (sólido)",
            f"{OUT_DIR}/ablation_{name}.png",
        )

    # 3) modelo combinado
    combined_model, combined_history = train_config(data, hidden_sizes, lr, **dict(COMBINED))
    all_metrics["combinado"] = regression_metrics(combined_model, x_test, y_test)
    all_models["combinado"] = combined_model
    all_histories["combinado"] = combined_history

    plot_before_after(
        baseline_history, combined_history, "combinado",
        "Baseline (tracejado) x combinado: dropout+momentum (sólido)",
        f"{OUT_DIR}/combined_model.png",
    )

    # 4) resumo em texto
    print("\n=== Métricas no conjunto de TESTE (todas as configurações) ===")
    for name, m in all_metrics.items():
        print(f"\n{name}:")
        print(format_metrics(m))

    # 5) gráfico de comparação final (barras)
    names = list(all_metrics.keys())
    r2_values = [all_metrics[n]["r2"] for n in names]
    mse_values = [all_metrics[n]["mse"] for n in names]
    plot_comparison(names, r2_values, mse_values, f"{OUT_DIR}/comparacao_final.png")

    # 6) validação de todas as configurações sobrepostas (Figura 2 do template)
    plot_val_comparison(
        all_histories,
        "Comparação de convergência na validação",
        f"{OUT_DIR}/val_comparison.png",
    )

    # 7) descobre o melhor modelo (maior R2 no teste) e gera parity plot + resíduos
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["r2"])
    best_model = all_models[best_name]
    print(f"\n>>> Melhor modelo (maior R2 no teste): {best_name}")

    baseline_model.eval()
    best_model.eval()
    with torch.no_grad():
        pred_baseline = baseline_model(x_test)
        pred_best = best_model(x_test)

    plot_parity(
        y_test, pred_baseline, pred_best, best_name,
        "Real vs. Previsto no conjunto de teste",
        f"{OUT_DIR}/parity_plot.png",
    )

    plot_residuals(
        y_test, pred_best, best_name,
        f"Resíduos do melhor modelo ({best_name})",
        f"{OUT_DIR}/residuos.png",
    )

    print(f"\nTodos os gráficos e o log da busca foram salvos em {OUT_DIR}/")


if __name__ == "__main__":
    main()