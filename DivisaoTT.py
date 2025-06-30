import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Carregar o arquivo CSV
df = pd.read_csv('BaseLabelEncoder.csv')  # Substitua pelo nome correto se for diferente

# Remover espaços nos nomes das colunas (boas práticas)
df.columns = df.columns.str.strip()

# Separar atributos (X) e rótulo (y)
X_BaseLabelEncoder = df.drop('Type of Answer', axis=1)
y_BaseLabelEncoder = df['Type of Answer']

# Divisão treino/teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    X_BaseLabelEncoder, y_BaseLabelEncoder, test_size=0.15, random_state=0
)

# Salvar as variáveis em um arquivo .pkl
with open('BaseTTLabelEncoder.pkl', mode='wb') as f:
    pickle.dump([X_treinamento, y_treinamento, X_teste, y_teste], f)


#####################################################################################

# Carregar o arquivo CSV
df = pd.read_csv('baseextratree.csv')  # Substitua pelo nome correto se for diferente

# Remover espaços nos nomes das colunas (boas práticas)
df.columns = df.columns.str.strip()

# Separar atributos (X) e rótulo (y)
X_BaseLabelEncoder = df.drop('Type of Answer', axis=1)
y_BaseLabelEncoder = df['Type of Answer']

# Divisão treino/teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    X_BaseLabelEncoder, y_BaseLabelEncoder, test_size=0.15, random_state=0
)

# Salvar as variáveis em um arquivo .pkl
with open('BaseTTExtraTree.pkl', mode='wb') as f:
    pickle.dump([X_treinamento, y_treinamento, X_teste, y_teste], f)


