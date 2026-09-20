"""
train.py
--------
Loop de treino com SGD puro (torch.optim.SGD, sem momentum) e MSELoss.
Os parâmetros momentum/weight_decay/l1_lambda existem aqui com valor 0
por padrão porque essa mesma função vai ser reaproveitada nos estudos de
ablação (L1, L2, dropout, momentum) depois.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_model(
    model,
    x_train, y_train,
    x_val, y_val,
    lr=0.05,
    epochs=1500,        # quantas vezes a rede vai passar pelo dataset de treino inteiro
    batch_size=8,       # de quantos em quantos exemplos os pesos são atualizados
    momentum=0.0,       
    weight_decay=0.0,   
    l1_lambda=0.0,       
    seed=42,
    verbose_every=0,
):
    torch.manual_seed(seed) #fixa

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()    # função de erro: mede o quão longe a previsão ficou do valor real
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)    # optimizer = quem de fato atualiza os pesos, usando o gradiente calculado no backward()


    history = {"train_loss": [], "val_loss": []} # erro diminuindo a cada época, até estabilizar.

    for epoch in range(1, epochs + 1):  #passada completa pelo dataset de treino
        model.train()  # Dropout fica ATIVO aqui
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:   # percorre o treino em fatias (batches)
            optimizer.zero_grad()             # 1. zera gradientes antigos
            pred = model(xb)                  # 2. forward pass
            loss = criterion(pred, yb)        # 3. erro (MSE)

            if l1_lambda > 0.0:
                l1_term = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_term

            loss.backward()                   # 4. backpropagation
            optimizer.step()                  # 5. atualiza os pesos

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches

        model.eval()  # Dropout fica DESLIGADO aqui
        with torch.no_grad():   # validação só mede o erro, não precisa calcular gradiente
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose_every and epoch % verbose_every == 0:
            print(f"epoch {epoch:4d}/{epochs} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}")

    return history  # devolve as duas curvas (treino e validação)


if __name__ == "__main__":  # "python src/train.py" pra treinar o baseline
    # e ver se o val MSE cai ao longo das época
    from data import prepare_data
    from model import MLP

    data = prepare_data("data/dataset_projeto1.csv")
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    model = MLP(hidden_sizes=(64, 64))
    history = train_model(
        model, x_train, y_train, x_val, y_val,
        lr=0.05, epochs=6000, batch_size=8, #aleatório para testar, não é o melhor hiperparâmetro
        verbose_every=1000,
    )
    print("\nMenor val MSE encontrado:", min(history["val_loss"]))