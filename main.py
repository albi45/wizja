from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.metrics import roc_curve
import shutil


def prepare_dataset(path, data_path, save_path):
    bad_files = []
    empty_pos = []
    bboxes = []
    next_bbs = False
    ready = False
    number_to_read = 0
    number = 0
    with open(path, "r") as f:
        for line in f:
            if ".jpg" in line:
                name = line.rstrip()
                print(f"File {data_path + name}")
                next_bbs = True
            elif next_bbs:
                number_to_read = int(line.rstrip())
                next_bbs = False
            elif number_to_read > 0:
                values = line.split(" ")
                bboxes.append((int(values[0]), int(values[1]), int(values[2]), int(values[3])))
                number_to_read -= 1
                if number_to_read == 0:
                    ready = True
            if ready:
                for bbox in bboxes:
                    img = cv2.imread(data_path + name)
                    img_neg, found = get_negative(img, bbox, bboxes)
                    if found:
                        try:
                            img_pos = img[bbox[1]:bbox[1] + bbox[3],
                                          bbox[0]:bbox[0] + bbox[2]]
                            img_pos = resize(img_pos)
                            img_neg = resize(img_neg)
                            cv2.imwrite(f"{save_path}0/{number}.jpg", img_neg)
                            cv2.imwrite(f"{save_path}1/{number}.jpg", img_pos)
                        except:
                            empty_pos.append(name + "_face_number-" + str(number_to_read))
                    else:
                        bad_files.append(name)
                    number += 1
                    print(f"Done number {number}")
                bboxes = []
                ready = False

    with open("bad_files.txt", "w") as f:
        for bad in bad_files:
            print(f"{bad}\n", f)
    with open("empty_pos.txt", "w") as f:
        for empty in empty_pos:
            print(f"{empty}\n", f)
    val_test_split()


def resize(img):
    if img.shape[0] > 25 or img.shape[1] > 25:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    img = cv2.resize(img, (25, 25), interpolation=interpolation)
    img = np.resize(img, (1, 25, 25, 1))
    return img


def get_negative(image, bbx, bbxs):
    stop = True
    trying = 10000
    while stop and trying > 0:
        x = np.random.randint(image.shape[0])
        y = np.random.randint(image.shape[1])
        stop = False
        for box in bbxs:
            if x < box[1] < x+bbx[3] and y < box[0] < y+bbx[2]:
                if (x+bbx[3]-box[1])*(y+bbx[2]-box[0]) > 0.25*box[2]*box[3]:
                    stop = True
        trying -= 1
    if trying < 0 and stop:
        found = False
    else:
        neg = image[x:x + bbx[3], y:y + bbx[2]]
        found = True
    return neg, found


def val_test_split():
    np.random.seed(42)
    directories = [f"dataset/val/1", f"dataset/val/0"]
    for directory in directories:
        folder = directory.split("/")[-1]
        files = list(os.listdir(directory))
        half = int(len(files)/2)
        np.random.shuffle(files)
        files = files[:half]
        for file in files:
            src = f"{directory}/{file}"
            shutil.move(src, f"dataset/test/{folder}/{file}")



def load_dataset():
    np.random.seed(42)
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
            img = np.resize(img, (25, 25, 1))
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


def detect(frame, cascade, model):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = cascade.detectMultiScale(frame_gray, 1.05, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        result = predict(model, frame_gray[y:y+h, x:x+w])
        if result == 1:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Capture - Face detection', frame)
    # cv2.waitKey(0)


def cascade_detect():
    model = keras.models.load_model('./best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # image = cv2.imread("WIDER_train/images/0--Parade/0_Parade_Parade_0_343.jpg")
    # detect(image, face_cascade, model)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame, Break!')
            break
        detect(frame, face_cascade, model)
        if cv2.waitKey(10) == 27:
            break


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
    # dirs = ["wider_face_split/wider_face_train_bbx_gt.txt", "wider_face_split/wider_face_val_bbx_gt.txt"]
    # prepare_dataset(dirs[0], "WIDER_train/images/", "dataset/train/")
    # prepare_dataset(dirs[1], "WIDER_val/images/", "dataset/val/")
    # train, val, test = load_dataset()
    # hist = training(train, val, 100)
    # plot_hist(hist)
    # evaluate(test)
    cascade_detect()

if __name__ == "__main__":
    main()
