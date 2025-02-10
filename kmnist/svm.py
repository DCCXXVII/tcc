import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import reciprocal, uniform
from torchvision import datasets, transforms

# Parâmetros
img_rows, img_cols = 28, 28
num_classes = 10

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

x_train, x_test, y_train, y_test = create_datasets(dataset_name="kmnist")

# # Criar subconjunto para testes rápidos
# sample_size = 1000  # Número de amostras reduzidas
# x_train = x_train[:sample_size]
# y_train = y_train[:sample_size]
# x_test= x_test[:sample_size]
# y_test = y_test[:sample_size]

t0 = time.time()

# # Pipeline do modelo
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svc', SVC())
# ])

# # Espaço dos hiperparâmetros
# param_distributions = {
#     'svc__C': uniform(1, 10),
#     'svc__kernel': ['rbf'],
#     'svc__gamma': reciprocal(0.001, 0.1) #['scale']
# }

# random_search = RandomizedSearchCV(
#     estimator=pipeline,
#     param_distributions=param_distributions,
#     n_iter=10,      # aumentar para melhores resultados (mais demorado)
#     #cv=5,          # aumentar para melhores resultados (mais demorado)
#     verbose=1,
#     random_state=123,
#     n_jobs=8
# )

# # Treinamento do modelo buscando os melhores hiperparâmetros
# random_search.fit(x_train, y_train)

# # Melhores hiperparâmetros
# print("Melhores hiperparâmetros:")
# print(random_search.best_params_)

# # Cálculo das métricas de avaliação
# y_test_pred = random_search.best_estimator_.predict(x_test)

C = 10.572971700825061
gamma = 0.01381957639278581
    
svm_clf = SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape="ovr", degree=3, gamma=gamma,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=True)

svm_clf.fit(x_train, y_train)
y_test_pred = svm_clf.predict(x_test)

print("Classification report:")
print(classification_report(y_test, y_test_pred, digits=3))

accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy:.3f}")

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Matriz de confusão:")
print(conf_matrix)

ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot()
plt.show()

run_time = time.time() - t0
print("Example run in %.3f s" % run_time)
plt.show()