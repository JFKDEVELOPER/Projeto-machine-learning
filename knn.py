# Carregar os dados salvos da base BaseLabelEncoder
import pickle

with open('BaseTTExtraTree.pkl', 'rb') as f:
    X_treinamento, y_treinamento, X_teste, y_teste = pickle.load(f)

print("=" * 50)
print("Dimensões dos conjuntos")
print("=" * 50)
print(f"X_treinamento: {X_treinamento.shape}")
print(f"y_treinamento: {y_treinamento.shape}")
print(f"X_teste:       {X_teste.shape}")
print(f"y_teste:       {y_teste.shape}")

# Treinar o modelo k-NN com k=2
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_treinamento, y_treinamento)

# Realizar previsões
previsoes = knn.predict(X_teste)

print("\n" + "=" * 50)
print("Comparação entre Previsões e Valores Reais")
print("=" * 50)
for i in range(len(previsoes)):
    print(f"Previsão: {previsoes[i]}  |  Real: {y_teste.values[i]}")

# Avaliar desempenho
from sklearn.metrics import accuracy_score, classification_report

acc = accuracy_score(y_teste, previsoes)
print("\n" + "=" * 50)
print("Resultados de Avaliação")
print("=" * 50)
print(f"Acurácia: {acc:.2%}")  # em formato percentual

print("\nRelatório de Classificação:\n")
print(classification_report(y_teste, previsoes, digits=4))

print(y_teste.value_counts())
