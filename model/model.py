from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
import os
import cv2

from utils.common import resize
from consts.common import SEED


def load_dataset():
    np.random.seed(SEED)
    train = []
    val = []
    test = []
    directories = [f"dataset/train/1", f"dataset/train/0", f"dataset/val/1", f"dataset/val/0", f"dataset/test/1", f"dataset/test/0"]
    for directory in directories:
        for nr, filename in enumerate(sorted(list(os.listdir(directory)))):
            if nr % 10:
                print(filename)
                print(nr)
            file = os.path.join(directory, filename)
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = np.resize(img, (25, 25, 1))
            label = directory.split("/")[-1]
            if directory.split("/")[-2] == "train":
                train.append((img, int(label)))
            elif directory.split("/")[-2] == "val":
                val.append((img, int(label)))
            else:
                test.append((img, int(label)))
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)
    x_train, y_train = zip(*train)
    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    train = (x_train, y_train)
    x_val, y_val = zip(*val)
    x_val = np.array(x_val, dtype=float)
    y_val = np.array(y_val, dtype=float)
    val = (x_val, y_val)
    x_test, y_test = zip(*test)
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=float)
    test = (x_test, y_test)
    return train, val, test


def get_model():
    model = keras.Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    far = tf.keras.metrics.FalsePositives(name="FP")
    frr = tf.keras.metrics.FalseNegatives(name="FN")
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer="Adam", loss=loss, metrics=["accuracy", loss, far, frr])
    return model


def training(train, val, epochs):
    model = get_model()
    x_train, y_train = train
    x_val, y_val = val
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3),
                    tf.keras.callbacks.ModelCheckpoint(filepath='./best_model.h5'),
                    tf.keras.callbacks.TensorBoard(log_dir='./logs'), ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, callbacks=my_callbacks)
    return history


def predict(model, x):
    x = resize(x)
    result = model.predict(x)
    result = np.where(result < 0.5, 0, 1)
    return result[0]


def plot_hist(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train", "test"])
    plt.show()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(["train", "test"])
    plt.show()

def evaluate(data):
    (x, y) = data
    acc = 0
    far = 0
    frr = 0
    length = x.shape[0]
    model = keras.models.load_model('./best_model.h5', compile=False)
    results = model.predict(x)
    results = np.where(results < 0.5, 0, 1)
    for i in range(0, length):
        if results[i] == y[i]:
            acc += 1
        if results[i] == 1 and y[i] == 0:
            far += 1
        if results[i] == 0 and y[i] == 1:
            frr += 1
    fpr, tpr, threshold = roc_curve(y, results, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(f"Accuracy = {acc/length:0.2f}")
    print(f"FAR = {far/length:0.2f}")
    print(f"FRR = {frr/length:0.2f}")
    print(f"EER = {eer:0.2f}")


def main():
    train, val, test = load_dataset()
    hist = training(train, val, 100)
    plot_hist(hist)
    evaluate(test)


if __name__ == '__main__':
    main()
