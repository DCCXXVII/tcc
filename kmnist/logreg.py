import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Parâmetros
img_rows, img_cols = 28, 28
num_classes = 10
train_samples = 10000

t0 = time.time()

# Carregamento e pré-processamento dos dados
def load_kmnist_data():
    def load(f):
        return np.load(f)['arr_0']

    x_train = load('kmnist/data/kmnist-train-imgs.npz')
    x_test = load('kmnist/data/kmnist-test-imgs.npz')
    y_train = load('kmnist/data/kmnist-train-labels.npz')
    y_test = load('kmnist/data/kmnist-test-labels.npz')

    x_train = x_train.reshape(-1, img_rows * img_cols)
    x_test = x_test.reshape(-1, img_rows * img_cols)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test, y_train, y_test

def create_datasets(dataset_name="kmnist"):
    """
    Cria os conjuntos de dados para MNIST ou Kuzushiji-MNIST.

    Parâmetros:
        dataset_name: "mnist" ou "kmnist"
    """
    if dataset_name.lower() == "mnist":
        # Carrega o MNIST usando torchvision
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Normalização padrão para MNIST
        ])

        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        # Converte os dados para numpy arrays
        x_train = train_dataset.data.numpy().reshape(-1, img_rows * img_cols)
        x_test = test_dataset.data.numpy().reshape(-1, img_rows * img_cols)
        y_train = train_dataset.targets.numpy()
        y_test = test_dataset.targets.numpy()

        # Normaliza os dados
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    elif dataset_name.lower() == "kmnist":
        # Carrega o KMNIST manualmente
        x_train, x_test, y_train, y_test = load_kmnist_data()

    else:
        raise ValueError("Parâmetro deve ser 'mnist' ou 'kmnist'.")

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = create_datasets(dataset_name="mnist")

t0 = time.time()

clf = LogisticRegression(C=50.0 / train_samples, penalty="l2", solver="saga", tol=0.1)
clf.fit(x_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100

y_pred= clf.predict(x_test)

# Cálculo das métricas de avaliação
print("Classification report:")
print(classification_report(y_test, y_pred, digits=3))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:")
print(conf_matrix)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l2_plot = plt.subplot(2, 5, i + 1)
    l2_plot.imshow(
        coef[i].reshape(28, 28),
        interpolation="nearest",
        cmap=plt.cm.RdBu,
        vmin=-scale,
        vmax=scale,
    )
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())
    l2_plot.set_xlabel("Class %i" % i)
plt.suptitle("Classification vector for...")

run_time = time.time() - t0
print("Example run in %.3f s" % run_time)
plt.show()