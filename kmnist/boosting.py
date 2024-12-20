import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

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

