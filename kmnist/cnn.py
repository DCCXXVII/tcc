import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from early_stopping_pytorch import EarlyStopping

# Utilizar GPU no treinamento da CNN
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device('cuda')
else:
    print("No GPU available. Training will run on CPU.")

# Parâmetros da CNN
img_rows, img_cols = 28, 28
num_classes = 10
batch_size = 128
epochs = 50
patience = 10

# Carregamento e pré-processamento dos dados
def load(f):
    return np.load(f)['arr_0']

def create_datasets(batch_size):
    x_train = load('kmnist/data/kmnist-train-imgs.npz')
    x_test = load('kmnist/data/kmnist-test-imgs.npz')
    y_train = load('kmnist/data/kmnist-train-labels.npz')
    y_test = load('kmnist/data/kmnist-test-labels.npz')

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # Porcentagem do conjunto de treino a ser utilizado para validação
    valid_size = 0.2

    # Obtendo os índices para validação
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # Sampler para obter os batches de treino e validação
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=0)
    
    valid_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=0)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=0)
    
    return train_loader, test_loader, valid_loader

train_loader, test_loader, valid_loader = create_datasets(batch_size)

# Definição da CNN
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Primeira camada convolucional: 1 input channel, 32 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Segunda camada convolucional: 32 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layer: 64*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(64 * 7 * 7, num_classes) # 28/4 = 7 (a dimensão da imagem 28x28 é dividida pela metade duas vezes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # Flatten
        x = self.fc1(x)
        return x

# Chamada do modelo construído
model = CNN(num_classes).to(device)

# Otimizador e critério a ser utilizado no cálculo da loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Código abaixo de https://github.com/Bjarten/early-stopping-pytorch/blob/main/MNIST_Early_Stopping_example.ipynb
def train_model(model, batch_size, patience, n_epochs):
    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [], [], [] ,[]
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        model.train()
        for batch, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        model.eval() # prep model for evaluation
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))
    
    if os.path.exists('checkpoint.pt'):
        os.remove('checkpoint.pt')
    else:
        print("File does not exist.")

    return  model, avg_train_losses, avg_valid_losses

# Treinamento do modelo
model, train_loss, valid_loss = train_model(model, batch_size, patience, epochs)

# Previsões do modelo
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        scores = model(x)
        preds = scores.argmax(dim=1)

        y_true.extend(y.cpu().numpy()) 
        y_pred.extend(preds.cpu().numpy()) 

# Visualização da perda
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 0.5)
plt.xlim(0, len(train_loss)+1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Cálculo das métricas de avaliação
print("Classification report:")
print(classification_report(y_true, y_pred, digits=3))

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")

conf_matrix = confusion_matrix(y_true, y_pred)
print("Matriz de confusão:")
print(conf_matrix)