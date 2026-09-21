# Projeto 1: Regressão com MLP e Estudo de Ablação

Disciplina: Introdução às Redes Neurais Artificiais - UNIFESP
Autora: Aline Breda Kuster

## Sobre o projeto

Implementação de uma rede neural MLP (Multi-Layer Perceptron) em PyTorch para
um problema de regressão, com:

- divisão do dataset em treino (10%), validação (10%) e teste (80%);
- um baseline "vanilla" (SGD puro, sem momentum/Adam/regularização), com
  arquitetura e learning rate escolhidos por busca empírica (menor MSE de
  validação);
- 4 estudos de ablação isolados: L1, L2, Dropout e Momentum, cada um com o
  próprio hiperparâmetro (λ, p ou β) escolhido por uma busca de validação
  dedicada, nunca pelo resultado de teste;
- um modelo combinado (Dropout + Momentum);
- avaliação final no conjunto de teste com MAE, MSE, RMSE e R², feita uma
  única vez, depois de todas as escolhas de hiperparâmetro já tomadas.

## Estrutura dos arquivos

| Arquivo | O que faz |
|---|---|
| `src/data.py` | Carrega o CSV, separa treino/validação/teste e padroniza os dados |
| `src/model.py` | Define a arquitetura da rede (MLP) |
| `src/train.py` | Loop de treino (SGD, MSELoss) |
| `src/metrics.py` | Calcula MAE, MSE, RMSE e R² |
| `src/plots.py` | Gera os gráficos (curvas de treino/validação, comparações, parity plot, resíduos) |
| `src/experiments.py` | Roda tudo: busca do baseline, busca do hiperparâmetro de cada ablação (por validação), treino final, as 4 ablações, o modelo combinado e todos os gráficos |

## Como rodar

```bash
pip install torch numpy pandas matplotlib
python src/experiments.py
```

Isso vai imprimir o progresso no terminal, salvar os gráficos e os logs das
buscas na pasta `outputs/` (`baseline_search_log.csv` para a arquitetura/lr do
baseline, `ablation_search_log.csv` para o hiperparâmetro de cada ablação), e
mostrar as métricas finais de cada configuração.

## Escolha dos hiperparâmetros

Tanto a arquitetura/lr do baseline quanto o λ (L1/L2), p (Dropout) e β
(Momentum) de cada ablação foram escolhidos pelo menor MSE de validação
atingido durante o treino - o conjunto de teste (240 pontos) só é usado uma
vez, na avaliação final de cada configuração já decidida, para preservá-lo
como medida honesta de generalização.

## Resultados principais (conjunto de teste)

| Modelo / Variação | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Baseline Vanilla | 0.5229 | 0.7231 | 0.5644 | -0.014 |
| + Dropout (p=0.1) | 0.4197 | 0.6479 | 0.5155 | 0.186 |
| + L2 (λ=1e-6) | 0.5142 | 0.7171 | 0.5621 | 0.003 |
| + L1 (λ=1e-7) | 0.5334 | 0.7303 | 0.5780 | -0.035 |
| + Momentum (β=0.7) | 0.4479 | 0.6693 | 0.5356 | 0.131 |
| Combinado (Dropout+Momentum) | 0.4881 | 0.6986 | 0.5507 | 0.053 |

Baseline escolhido pela busca empírica: arquitetura `(64,64)`, `lr=0.1`.
Dropout (p=0.1) foi a técnica isolada mais eficaz; combinar Dropout com
Momentum não superou o Dropout isolado, mostrando que as duas técnicas não
são simplesmente aditivas.
