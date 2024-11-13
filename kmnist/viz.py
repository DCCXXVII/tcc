from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fontsize = 50
font = ImageFont.truetype('kmnist/font/NotoSansJP-VariableFont_wght.ttf', fontsize, encoding='utf-8')

# Mapeamento dos caracteres e seus índices (classes)
unicode_map = {index: char for index, codepoint, char in pd.read_csv('kmnist/kmnist_classmap.csv').values}
train_imgs = np.load('kmnist/data/kmnist-train-imgs.npz')["arr_0"]
train_labels = np.load('kmnist/data/kmnist-train-labels.npz')["arr_0"]

classes = [0, 2] # Escolha das classes a serem visualizadas
fig, axes = plt.subplots(len(classes), 6, figsize=(10, 6))

for row, class_label in enumerate(classes):
    # Imagem moderna pra cada classe escolhida
    char = unicode_map[class_label]
    img = Image.new('RGB', (80, 80), color='black')
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(char, font=font)
    draw.text(((80 - w) / 2, (80 - h) / 2), char, font=font, fill='white')
    
    # Plot da imagem moderna na primeira coluna do output
    axes[row, 0].imshow(np.array(img))
    axes[row, 0].axis('off')

    # Pega 5 imagens aleatórias do conjunto para cada classe
    class_indices = np.where(train_labels == class_label)[0]
    n = np.random.randint(len(class_indices), size=5)
    sample_images = train_imgs[class_indices[n]]

    for col, img in enumerate(sample_images, start=1):
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()