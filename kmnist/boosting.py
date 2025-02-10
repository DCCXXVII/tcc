import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from torchvision import datasets, transforms
import xgboost as xgb
import optuna

# Parâmetros
img_rows, img_cols = 28, 28
num_classes = 10

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

# Criar subconjunto para testes rápidos
sample_size = 1000  # Número de amostras reduzidas
x_train_small = x_train[:sample_size]
y_train_small = y_train[:sample_size]
x_test_small = x_test[:sample_size]
y_test_small = y_test[:sample_size]

# Define the objective function for Optuna
def objective(trial):
    params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method': 'auto',
        'verbosity': 0
    }

    dtrain = xgb.DMatrix(x_train_small, label=y_train_small)
    dtest = xgb.DMatrix(x_test_small, label=y_test_small)

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)
    preds = model.predict(dtest)
    accuracy = accuracy_score(y_test_small, preds)
    return accuracy

# # Melhores hiperparâmetros pelo Optuna
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, show_progress_bar=True, n_trials=50)
# best_params = study.best_params
# print(f"Best hyperparameters: {best_params}")

# Para MNIST

# Para K-MNIST
best_params = {
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'objective': 'multi:softmax',
    'num_class': num_classes,
    'eval_metric': 'mlogloss',
    'booster': 'dart',
    'lambda': 5.12281677965647e-08,
    'alpha': 6.866809111671214e-08,
    'max_depth': 7,
    'eta': 0.35632071788449393,
    'gamma': 4.5333397884416503e-07,
    'grow_policy': 'depthwise',
    'subsample': 0.4199840920892194,
    'colsample_bytree': 0.5979597904388615,
    'min_child_weight': 1,
    'tree_method': 'auto',
    'verbosity': 0
}
print(f"Best hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

final_model = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)

# Evaluate the final model
final_preds = final_model.predict(dtest)
final_preds = final_preds.astype(int)  # Converte para inteiros
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Final model accuracy: {final_accuracy}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, final_preds))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.show()