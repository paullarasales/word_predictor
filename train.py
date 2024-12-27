import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import LearningRateScheduler, TensorBoard
import math
import os
import csv

DATA_FILE = "expanded_scrambled_words_dataset.csv"
LOG_DIR = "logs"
EPOCHS = 100
BATCH_SIZE = 32
EMBEDDING_DIM = 50
LSTM_UNITS = 128
VALIDATION_SPLIT = 0.2

def lr_schedule(epoch):
    initial_lr = 0.001
    decay_rate = 0.1
    decay_steps = 10
    lr = initial_lr * math.exp(-decay_rate * (epoch / decay_steps))
    return float(lr) 

data = []
labels = []
char_to_index = {chr(i): i - ord('a') + 1 for i in range(ord('a'), ord('z') + 1)}
char_to_index["<PAD>"] = 0

max_length = 0

with open(DATA_FILE, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        scrambled, word = row
        data.append(scrambled)
        labels.append(word)
        max_length = max(max_length, len(scrambled), len(word))

X = np.array([[char_to_index[char] for char in word] + [0] * (max_length - len(word)) for word in data])
y = np.array([[char_to_index[char] for char in word] + [0] * (max_length - len(word)) for word in labels])


model = Sequential([
    Embedding(input_dim=len(char_to_index), output_dim=EMBEDDING_DIM, input_length=max_length),
    LSTM(LSTM_UNITS, return_sequences=True),
    Dense(len(char_to_index), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

os.makedirs(LOG_DIR, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

lr_scheduler = LearningRateScheduler(lr_schedule)

X_train = X[:int(len(X) * (1 - VALIDATION_SPLIT))]
X_val = X[int(len(X) * (1 - VALIDATION_SPLIT)):]
y_train = y[:int(len(y) * (1 - VALIDATION_SPLIT))]
y_val = y[int(len(y) * (1 - VALIDATION_SPLIT)):]

model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback, lr_scheduler]
)

model.save("unscrambler_model.keras")
print("Model training complete and saved!")
