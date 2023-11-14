import os
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import librosa.display
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import librosa
import  warnings
import sounddevice as sd
from scipy.io.wavfile import write
import seaborn as sns
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")
import umap
import pickle
import plotly.graph_objects as go
from keras.utils import to_categorical
from sklearn.neighbors import NearestNeighbors

data_dir = './train/data_set/'
spkr_id_file = './data_lists/speaker_id.scp'

spkr_id = {}

with open(spkr_id_file, "r", encoding = 'utf-8') as file:
    for line in file:
        key, value = line.strip().split(":", 1)
        spkr_id[int(key)] = value

def read_data(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    
    data, labels = [], []
    for line in lines:
        path, label = line.strip().split(":")
        wav_data, _ = librosa.load(path, sr=16000)  # Load the audio file using librosa
        data.append(wav_data)
        labels.append(int(label))

    return np.array(data), np.array(labels)  # Convert the list of arrays and labels to a numpy array

test_data, test_labels = read_data('./data_lists/test.scp')

# Expanding the dimension of the data
test_data = np.expand_dims(test_data, axis=-1)

num_classes = len(set(spkr_id))

test_labels = to_categorical(test_labels, num_classes = num_classes)

load_model = tf.keras.models.load_model("final-model")
y_pred = load_model.predict(test_data)

y_pred_labels = [np.argmax(y_pred[i]) for i in range(len(y_pred))]

# Получение представлений перед последним слоем
model_without_last_layer = tf.keras.models.Model(inputs=load_model.inputs, outputs=load_model.layers[-2].output)
features = model_without_last_layer.predict(test_data)

# Применение UMAP
reducer = umap.UMAP(spread=3.0)
embedding = reducer.fit_transform(features)

test_labels_multiclass = np.argmax(test_labels, axis=1)

# Загрузка файла и создание словаря
spkr_id_dict = {}
with open('./data_lists/speaker_id.scp', 'r', encoding = 'utf-8') as f:
    for line in f:
        id, name = line.strip().split(':')
        spkr_id_dict[int(id)] = name

# Замена ID класса на имена в метках классов
class_names = [spkr_id_dict[i] for i in np.argmax(test_labels, axis=1)]

# Преобразование в numpy массив для удобства
features_np = np.array(features)

# Define the number of neighbors for KNN
k = 100

# Fit the model
nbrs = NearestNeighbors(n_neighbors=k).fit(features_np)

# Get distances and indices of k neighbors from the fitted model
distances, indices = nbrs.kneighbors(features_np)

# Define the threshold as the 95th percentile of the mean distances
threshold = np.percentile(np.mean(distances, axis=1), 70)

# Identify outliers
outliers = np.mean(distances, axis=1) > threshold

# Remove outliers
features_clean = features_np[~outliers]
test_labels_clean = test_labels_multiclass[~outliers]

# Применение UMAP к очищенным данным
embedding_clean = reducer.fit_transform(features_clean)

# Визуализация
fig = go.Figure(data=go.Scatter(
    x=embedding_clean[:, 0],
    y=embedding_clean[:, 1],
    mode='markers',
    marker=dict(
        size=5,
        color=test_labels_clean, #set color equal to a variable
        colorscale='Spectral', # one of plotly colorscales
        showscale=True
    ),
    text=[f'Class: {spkr_id_dict[name]}' for name in test_labels_clean], # метки классов
    hoverinfo='text'
))

fig.update_layout(title='speaker dataset', autosize=False,
                  width=1900, height=1000, # Здесь установите размеры вашего экрана
                  margin=dict(l=50, r=50, b=100, t=100, pad=10))

fig.show()

# # Импорт DecisionBoundaryDisplay
# from sklearn.inspection import DecisionBoundaryDisplay

# # Создание и отображение решающей границы
# disp = DecisionBoundaryDisplay.from_estimator(
#     load_model,
#     test_data.squeeze(),
#     response_method = 'predict',
#     alpha = 0.5,
# )
# disp.ax_.scatter(embedding_clean[:, 0], embedding_clean[:, 1], c=test_labels_clean, s=20, edgecolor="k")
# disp.ax_.set_title('Решающая граница изолирующего дерева')
# plt.show()