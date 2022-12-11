from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np


def read_data(path, data_path, save_path):
    # bad_files = ["File WIDER_train/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_531.jpg", "WIDER_train/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_114.jpg", "WIDER_train/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_319.jpg","WIDER_train/images/10--People_Marching/10_People_Marching_People_Marching_2_164.jpg", "WIDER_train/images/10--People_Marching/10_People_Marching_People_Marching_2_417.jpg", "WIDER_train/images/10--People_Marching/10_People_Marching_People_Marching_2_476.jpg", "WIDER_train/images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_445.jpg", "WIDER_train/images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_1000.jpg", "WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_545.jpg", "WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_722.jpg", "WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_730.jpg", "WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_333.jpg", "WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_515.jpg", "WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_254.jpg", "WIDER_train/images/0--Parade/0_Parade_Parade_0_52.jpg", "WIDER_train/images/0--Parade/0_Parade_Parade_0_179.jpg", "WIDER_train/images/0--Parade/0_Parade_marchingband_1_237.jpg", "WIDER_train/images/0--Parade/0_Parade_Parade_0_926.jpg", "WIDER_train/images/0--Parade/0_Parade_marchingband_1_1031.jpg", "WIDER_train/images/0--Parade/0_Parade_marchingband_1_615.jpg", "WIDER_train/images/0--Parade/0_Parade_Parade_0_570.jpg", "WIDER_train/images/0--Parade/0_Parade_Parade_0_939.jpg", "WIDER_train/images/0--Parade/0_Parade_Parade_0_763.jpg"]
    bad_files = []
    empty_pos = []
    bboxes = []
    names = []
    labels = []
    images = []
    next_bbs = False
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
            elif number_to_read == 0:
                for bbox in bboxes:
                    img = cv2.imread(data_path + name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_neg, found = get_negative(img, bbox, bboxes)
                    if found:
                        try:
                            img_pos = img[int(values[1]):int(values[1]) + int(values[3]),
                                          int(values[0]):int(values[0]) + int(values[2])]
                            if img_pos.shape[0] > 25 or img_pos.shape[1] > 25:
                                interpolation = cv2.INTER_AREA
                            else:
                                interpolation = cv2.INTER_LINEAR
                            img_pos = cv2.resize(img_pos, (25, 25), interpolation=interpolation)
                            img_neg = cv2.resize(img_neg, (25, 25), interpolation=interpolation)
                            cv2.imwrite(f"{save_path}0/{number}.jpg", img_neg)
                            cv2.imwrite(f"{save_path}1/{number}.jpg", img_pos)
                        except:
                            empty_pos.append(name + "_face_number-" + str(number_to_read))
                    else:
                        bad_files.append(name)
                    number += 1
                    print(f"Done number {number}")

    with open("bad_files.txt", "w") as f:
        for bad in bad_files:
            print(f"{bad}\n", f)
    with open("empty_pos.txt", "w") as f:
        for empty in empty_pos:
            print(f"{empty}\n", f)
    # for name in names:
    #     images.append(cv2.imread(data_path + name))
    return images, labels


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


def detect(frame, cascade):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (495, 177), (495+37, 177+51), (255, 0, 0), 2)
    cv2.imshow('Capture - Face detection', frame)
    cv2.waitKey(0)


def haar_detect():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread("WIDER_train/images/0--Parade/0_Parade_marchingband_1_5.jpg")
    detect(image, face_cascade)
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened:
    #     print('Error opening video capture')
    #     exit(0)
    # while True:
    #     ret, frame = cap.read()
    #     if frame is None:
    #         print('No captured frame, Break!')
    #         break
    #     detectAndDisplay(frame, face_cascade)
    #     if cv2.waitKey(10) == 27:
    #         break


def get_model():
    model = keras.Sequential([
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
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer="Adam", loss=loss, metrics=["accuracy"])
    return model


def training(train, val, epochs):
    model = get_model()
    x_train, y_train = train
    x_val, y_val = val
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=10),
                    tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5'),
                    tf.keras.callbacks.TensorBoard(log_dir='./logs'), ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, callbacks=my_callbacks)
    return history


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


def main():
    dirs = ["wider_face_split/wider_face_train_bbx_gt.txt", "wider_face_split/wider_face_val_bbx_gt.txt"]
    train_names, train_labels = read_data(dirs[0], "WIDER_train/images/", "dataset/train/")
    val_names, val_labels = read_data(dirs[1], "WIDER_val/images/", "dataset/val/")


if __name__ == "__main__":
    main()
