"""
model.py
--------
Definição da arquitetura MLP (Multi-Layer Perceptron).
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=1, hidden_sizes=(16, 8), out_features=1, dropout_p=0.0):
        super().__init__()
        '''Cria uma MLP com uma entrada, duas camadas e uma saída'''

        layers = []
        prev_size = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_size = h
        layers.append(nn.Linear(prev_size, out_features))  # saída, sem ativação
        # regressão: a saída precisa poder ser qualquer número real,
        # positivo ou negativo, então NÃO colocamos ReLU/sigmoid na saída

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
         # forward = roda o x pelo esqueleto montado no __init__ (self.net),
    # passando por cada camada em sequência, até virar a previsão de y
