import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import reciprocal, uniform

# Parâmetros
img_rows, img_cols = 28, 28
num_classes = 10

# Carregamento e pré-processamento dos dados
def load(f):
    return np.load(f)['arr_0']

def create_datasets():
    x_train = load('kmnist/data/kmnist-train-imgs.npz')
    x_test = load('kmnist/data/kmnist-test-imgs.npz')
    y_train = load('kmnist/data/kmnist-train-labels.npz')
    y_test = load('kmnist/data/kmnist-test-labels.npz')

    x_train = x_train.reshape(-1, img_rows * img_cols)
    x_test = x_test.reshape(-1, img_rows * img_cols)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = create_datasets()

# # # Gráfico do tempo de execução à medida que aumenta o sample_size
# # # Adicionar um tempo máximo de execução

# Criar subconjunto para testes rápidos
sample_size = 3000  # Número de amostras reduzidas
x_train_small = x_train[:sample_size]
y_train_small = y_train[:sample_size]
x_test_small = x_test[:sample_size]
y_test_small = y_test[:sample_size]

# Pipeline do modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Espaço dos hiperparâmetros
param_distributions = {
    'svc__C': uniform(1, 10),
    'svc__kernel': ['rbf'],
    'svc__gamma': reciprocal(0.001, 0.1) #['scale']
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,      # aumentar para melhores resultados (mais demorado)
    #cv=5,          # aumentar para melhores resultados (mais demorado)
    verbose=1,
    random_state=123,
    n_jobs=8
)

# Treinamento do modelo buscando os melhores hiperparâmetros
random_search.fit(x_train_small, y_train_small)

# Melhores hiperparâmetros
print("Melhores hiperparâmetros:")
print(random_search.best_params_)

# Cálculo das métricas de avaliação
y_test_pred = random_search.best_estimator_.predict(x_test_small)

print("Classification report:")
print(classification_report(y_test_small, y_test_pred, digits=3))

accuracy = accuracy_score(y_test_small, y_test_pred)
print(f"Accuracy: {accuracy:.3f}")

conf_matrix = confusion_matrix(y_test_small, y_test_pred)
print("Matriz de confusão:")
print(conf_matrix)

ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot()
plt.show()