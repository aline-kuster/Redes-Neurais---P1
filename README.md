# Projeto 1: Regressão com MLP e Estudo de Ablação

Disciplina: Introdução às Redes Neurais Artificiais — UNIFESP
Autora: Aline Breda Kuster

## Sobre o projeto

Implementação de uma rede neural MLP (Multi-Layer Perceptron) em PyTorch para
um problema de regressão, com:

- divisão do dataset em treino (10%), validação (10%) e teste (80%);
- um baseline "vanilla" (SGD puro, sem momentum/Adam/regularização), com
  arquitetura e learning rate escolhidos por busca empírica;
- 4 estudos de ablação isolados: L1, L2, Dropout e Momentum;
- um modelo combinado (Dropout + Momentum);
- avaliação final no conjunto de teste com MAE, MSE, RMSE e R².

## Estrutura dos arquivos

| Arquivo | O que faz |
|---|---|
| `src/data.py` | Carrega o CSV, separa treino/validação/teste e padroniza os dados |
| `src/model.py` | Define a arquitetura da rede (MLP) |
| `src/train.py` | Loop de treino (SGD, MSELoss) |
| `src/metrics.py` | Calcula MAE, MSE, RMSE e R² |
| `src/plots.py` | Gera os gráficos (curvas de treino/validação, comparações, parity plot, resíduos) |
| `src/experiments.py` | Roda tudo: busca do baseline, treino final, as 4 ablações, o modelo combinado e todos os gráficos |

## Como rodar

```bash
pip install torch numpy pandas matplotlib
python src/experiments.py
```

Isso vai imprimir o progresso no terminal, salvar os gráficos e o log da busca
na pasta `outputs/`, e mostrar as métricas finais de cada configuração.

## Resultados principais (conjunto de teste)

| Modelo / Variação | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Baseline Vanilla | 0.5229 | 0.7231 | 0.5644 | -0.014 |
| + Dropout (p=0.3) | 0.4796 | 0.6926 | 0.5511 | 0.070 |
| + L2 (λ=1e-5) | 0.5345 | 0.7311 | 0.5746 | -0.037 |
| + L1 (λ=1e-6) | 0.5268 | 0.7258 | 0.5669 | -0.022 |
| + Momentum (β=0.7) | 0.4479 | 0.6693 | 0.5356 | 0.131 |
| Combinado (Dropout+Momentum) | 0.4918 | 0.7013 | 0.5588 | 0.046 |

Baseline escolhido pela busca empírica: arquitetura `(64,64)`, `lr=0.1`.
