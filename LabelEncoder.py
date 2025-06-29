from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Lê o arquivo original
df = pd.read_csv("data.csv", sep=';', encoding='latin1')

# Converte para NumPy array
data = df.values

print("Antes da codificação (linha 0):", data[0])

# Cria os LabelEncoders
label_encoder_pais = LabelEncoder()
label_encoder_educacao = LabelEncoder()
label_encoder_disciplina = LabelEncoder()
label_encoder_area = LabelEncoder()
label_encoder_conceitos = LabelEncoder()
label_encoder_sexo = LabelEncoder()

# Aplica o LabelEncoder nas colunas categóricas
data[:, 1] = label_encoder_pais.fit_transform(data[:, 1])
data[:, 3] = label_encoder_sexo.fit_transform(data[:, 3])
data[:, 4] = label_encoder_educacao.fit_transform(data[:, 4])
data[:, 5] = label_encoder_disciplina.fit_transform(data[:, 5])
data[:, 6] = label_encoder_area.fit_transform(data[:, 6])
data[:, 7] = label_encoder_conceitos.fit_transform(data[:, 7])

print("Depois da codificação (linha 0):", data[0])

# Converte de volta para DataFrame usando os nomes originais das colunas
df_processado = pd.DataFrame(data, columns=df.columns)


# Salva o DataFrame processado em CSV (sem o índice)
df_processado.to_csv("base_processada.csv", index=False)


