from tensorflow import keras
import cv2
import time

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

def cascade_detect_wideo(filename=None, use_webcam=True):
    model = keras.models.load_model('../best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if use_webcam:
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
    else:
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened:
            print('Error loading video file')
            exit(0)
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('No captured frame, Break!')
                break
            detect(frame, face_cascade, model)
            if cv2.waitKey(10) == 27:
                break

def cascade_detect_video(filename=None, use_webcam=True, show_fps=False):
    model = keras.models.load_model('../best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened:
            print('Error opening video capture')
            exit(0)
    else:
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened:
            print('Error loading video file')
            exit(0)
    frames = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame, Break!')
            break
        detect(frame, face_cascade, model)
        frames += 1
        if cv2.waitKey(10) == 27:
            break
    end_time = time.time()
    duration = end_time - start_time
    fps = frames / duration
    if show_fps:
        print("FPS: {:.2f}".format(fps))

def anonymize_face(frame, cascade, model, single_shot=False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = cascade.detectMultiScale(frame_gray, 1.05, 3)
    for (x, y, w, h) in faces:
        # face_roi = frame[y:y + h, x:x + w]
        # blurred = cv2.medianBlur(face_roi, 99)
        # frame[y:y+h, x:x+w] = blurred
        result = predict(model, frame_gray[y:y+h, x:x+w])
        if result == 1:
            face_roi = frame[y:y + h, x:x + w]
            blurred = cv2.medianBlur(face_roi, 99)
            frame[y:y+h, x:x+w] = blurred
    cv2.imshow('Capture - Face detection', frame)
    if single_shot:
        cv2.waitKey(0)

def anonymize_on_video(filename=None, use_webcam=True, save_to_avi=False):
    model = keras.models.load_model('../best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if use_webcam:
        cap = cv2.VideoCapture(0)
        if save_to_avi:
            size = (int(cap.get(3)), int(cap.get(4)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            result_video = cv2.VideoWriter('results/anonymized_video.avi', 
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    fps, size)
        if not cap.isOpened:
            print('Error opening video capture')
            exit(0)
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('No captured frame, Break!')
                break
            if save_to_avi:
                anonymize_face(frame, face_cascade, model)
                result_video.write(frame)
            else:
                anonymize_face(frame, face_cascade, model)
            if cv2.waitKey(10) == 27:
                break
    else:
        cap = cv2.VideoCapture(filename)
        if save_to_avi:
            size = (int(cap.get(3)), int(cap.get(4)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            result_video = cv2.VideoWriter('results/anonymized_video.avi', 
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    fps, size)
        if not cap.isOpened:
            print('Error loading video file')
            exit(0)
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('No captured frame, Break!')
                break
            if save_to_avi:
                anonymize_face(frame, face_cascade, model)
                result_video.write(frame)
            else:
                anonymize_face(frame, face_cascade, model)
            if cv2.waitKey(10) == 27:
                cap.release()
                result_video.release()                    
                # Closes all the frames
                cv2.destroyAllWindows()
                break

def cascade_detect_image(filename, save_img=False, saved_img_path=None):
    model = keras.models.load_model('../best_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame = cv2.imread(filename)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray, 1.05, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        result = predict(model, frame_gray[y:y+h, x:x+w])
        if result == 1:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if save_img:
        cv2.imwrite(saved_img_path, frame)
    cv2.imshow('Faces', frame)
    if cv2.waitKey(1) == 27: # Escape
        cv2.destroyAllWindows()

def main():
    cascade_detect_image("../WIDER_train/images/0--Parade/0_Parade_Parade_0_343.jpg")


if __name__ == '__main__':
    main()