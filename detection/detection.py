from tensorflow import keras
import cv2

from model.model import predict


def detect(frame, cascade, model, single_shot=False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = cascade.detectMultiScale(frame_gray, 1.05, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        result = predict(model, frame_gray[y:y+h, x:x+w])
        if result == 1:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Capture - Face detection', frame)
    if single_shot:
        cv2.waitKey(0)


def cascade_detect_wideo():
    model = keras.models.load_model('./best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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


def cascade_detect_image(img_path):
    model = keras.models.load_model('./best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(img_path)
    detect(image, face_cascade, model, single_shot=True)


def main():
    cascade_detect_image("WIDER_train/images/0--Parade/0_Parade_Parade_0_343.jpg")


if __name__ == '__main__':
    main()