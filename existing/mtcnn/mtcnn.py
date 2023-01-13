import cv2
from mtcnn.mtcnn import MTCNN
import time


def mtcnn_detect_on_img(filename, save_img=False, saved_img_path=None):
    mtcnn_model = MTCNN()
    img = cv2.imread(filename)

    while not img is None:

        faces = mtcnn_model.detect_faces(img)# result
        #to draw faces on image
        for face in faces:
            x, y, w, h = face['box']
            x1, y1 = x + w, y + h
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        if save_img:
            cv2.imwrite(saved_img_path, img)
        cv2.imshow('face detected', img)
        if cv2.waitKey(1) == 27: # Escape
            cv2.destroyAllWindows()
            break

def mtcnn_anonymize_on_video(filename, save_to_avi=True, show_fps=False):
    mtcnn_model = MTCNN()
    cap = cv2.VideoCapture(filename)
    success, frame = cap.read()

    if save_to_avi:
        size = (int(cap.get(3)), int(cap.get(4)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter('results/anonymized_mtcnn.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, size)
        frames = 0
        start_time = time.time()
        while success:

            faces = mtcnn_model.detect_faces(frame)
            if faces != []:
                for face in faces:
                    traced_rectangle = face['box']
                    x, y, w, h = traced_rectangle
                    frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 99)

            result.write(frame)
            frames += 1
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) == 27: # Escape
                cap.release()
                result.release()                    
                cv2.destroyAllWindows()
                break
            success, frame = cap.read()
        end_time = time.time()
        duration = end_time - start_time
        fps = frames / duration
        if show_fps:
            print("FPS: {:.2f}".format(fps))
    else:
        frames = 0
        start_time = time.time()
        while success:
            faces = mtcnn_model.detect_faces(frame)
            if faces != []:
                for face in faces:
                    traced_rectangle = face['box']
                    x, y, w, h = traced_rectangle
                    frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 99)
            frames += 1
            cv2.imshow('face detected', frame)
            if cv2.waitKey(1) == 27: # Escape
                cv2.destroyAllWindows()
                break
            success, frame = cap.read()
        end_time = time.time()
        duration = end_time - start_time
        fps = frames / duration
        if show_fps:
            print("FPS: {:.2f}".format(fps))
