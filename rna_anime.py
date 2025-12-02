# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =====================================================
# 0) CRIAR PASTA DE SAÍDA
# =====================================================
if not os.path.exists("resultados"):
    os.makedirs("resultados")

# =====================================================
# FUNÇÃO PARA CONVERTER AS LISTAS DO CSV
# =====================================================
def parse_list(texto):
    try:
        valor = ast.literal_eval(texto)
        if isinstance(valor, list):
            return " ".join(valor)   # vira "Action Drama Fantasy"
        return str(texto)
    except:
        return str(texto)

# =====================================================
# 1) CARREGAR DATASET
# =====================================================
df = pd.read_csv("anime_dataset.csv")
print("Formato original:", df.shape)

# =====================================================
# 2) PROCESSAR GÊNEROS E ESTÚDIOS
# =====================================================
df['genres'] = df['genres'].apply(parse_list)
df['studios'] = df['studios'].apply(parse_list)

# Manter apenas colunas úteis
df = df[['score', 'episodes', 'members', 'year', 'genres', 'studios']]
df = df.dropna()

# One-hot encoding
df = pd.get_dummies(df, columns=['genres', 'studios'])

print("Formato após processamento:", df.shape)

# =====================================================
# 3) SEPARAR X E y
# =====================================================
y = df['score'].values
X = df.drop(['score'], axis=1)
colunas_X = X.columns.tolist()
X = X.values

# =====================================================
# 4) NORMALIZAR
# =====================================================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# =====================================================
# 5) TREINO / TESTE
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 6) MODELO MLP
# =====================================================
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # saída contínua

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

print(model.summary())

# =====================================================
# 7) TREINAMENTO
# =====================================================
history = model.fit(
    X_train,
    y_train,
    epochs=120,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =====================================================
# 8) GRÁFICO MSE
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.legend()
plt.xlabel("Épocas")
plt.ylabel("Erro MSE")
plt.title("Evolução do Erro MSE")
plt.savefig("resultados/treinamento_mse.png", dpi=300)
plt.close()

# =====================================================
# 9) AVALIAÇÃO
# =====================================================
pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"MSE: {mse:.4f}")
print(f"R²:  {r2:.4f}")

# =====================================================
# 10) GRÁFICO REAL vs PREDITO
# =====================================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred, s=10)
plt.xlabel("Score real")
plt.ylabel("Score predito")
plt.title("Real vs Predito")
plt.savefig("resultados/real_vs_predito.png", dpi=300)
plt.close()

# =====================================================
# 11) SALVAR MODELO E SCALER
# =====================================================
model.save("resultados/modelo_prever_score.h5")
joblib.dump(scaler, "resultados/scaler_score.joblib")
joblib.dump(colunas_X, "resultados/labels_X.joblib")

print("\nModelo e scaler salvos na pasta 'resultados/'")

# =====================================================
# 12) TESTE MANUAL (MENU INTERATIVO)
# =====================================================
def prever_score_terminal():

    print("\n==== TESTE MANUAL DE PREDIÇÃO DO SCORE ====\n")

    # carregar scaler e labels
    scaler = joblib.load("resultados/scaler_score.joblib")
    labels_X = joblib.load("resultados/labels_X.joblib")

    # separar grupos de colunas
    colunas_numericas = []
    colunas_genres = []
    colunas_studios = []

    for col in labels_X:
        if col.startswith("genres_"):
            colunas_genres.append(col)
        elif col.startswith("studios_"):
            colunas_studios.append(col)
        else:
            colunas_numericas.append(col)

    valores = []

    # ====================== NUMÉRICOS ======================
    print("\nDigite os valores numéricos:\n")
    for col in colunas_numericas:
        while True:
            try:
                v = float(input(f"{col}: "))
                valores.append(v)
                break
            except:
                print("Valor inválido.")

    # ====================== GÊNEROS ======================
    print("\n=== GÊNEROS DISPONÍVEIS ===")
    for i, col in enumerate(colunas_genres):
        print(f"{i+1}. {col.replace('genres_', '')}")

    entrada = input("\nEscolha os gêneros (ex: 1,4,7): ")
    vetor_genres = [0]*len(colunas_genres)

    if entrada.strip():
        try:
            indices = [int(x)-1 for x in entrada.split(",")]
            for idx in indices:
                if 0 <= idx < len(vetor_genres):
                    vetor_genres[idx] = 1
        except:
            pass

    valores.extend(vetor_genres)

    # ====================== ESTÚDIOS ======================
    print("\n=== ESTÚDIOS DISPONÍVEIS ===")
    for i, col in enumerate(colunas_studios):
        print(f"{i+1}. {col.replace('studios_', '')}")

    entrada = input("\nEscolha o(s) estúdio(s) (ex: 2,5): ")
    vetor_studios = [0]*len(colunas_studios)

    if entrada.strip():
        try:
            indices = [int(x)-1 for x in entrada.split(",")]
            for idx in indices:
                if 0 <= idx < len(vetor_studios):
                    vetor_studios[idx] = 1
        except:
            pass

    valores.extend(vetor_studios)

    # Montar vetor final
    vetor_final = np.array([valores])
    vetor_final = scaler.transform(vetor_final)

    predicted = model.predict(vetor_final).flatten()[0]

    print("\n===== RESULTADO =====")
    print(f"Score previsto: {predicted:.2f}\n")

    return predicted

# =====================================================
# 13) EXECUTAR TESTE MANUAL
# =====================================================
import sys
if len(sys.argv) > 1 and sys.argv[1] == "--teste":
    prever_score_terminal()
    exit()
