import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Parâmetros
img_rows, img_cols = 28, 28
num_classes = 10
k = 4


# Carregamento e pré-processamento dos dados
def load(f):
    return np.load(f)['arr_0']

def create_datasets():
    x_train = load('kmnist/data/kmnist-train-imgs.npz')
    x_test = load('kmnist/data/kmnist-test-imgs.npz')
    y_train = load('kmnist/data/kmnist-train-labels.npz')
    y_test = load('kmnist/data/kmnist-test-labels.npz')

    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = create_datasets()

# Treinamento do modelo
pca = PCA(n_components = 60)                                                # Grid search faria sentido aqui? (separando em treino/valid)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)    # Grid search faria sentido aqui? (separando em treino/valid)
knn.fit(x_train, y_train)

y_test_pred = knn.predict(x_test)

# Cálculo das métricas de avaliação
print("Classification report:")
print(classification_report(y_test, y_test_pred, digits=3))

test_score = knn.score(x_test, y_test)
print(f"Accuracy: {test_score:.3f}")

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Matriz de confusão:")
print(conf_matrix)