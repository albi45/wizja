import cv2
import numpy as np
import time


def ssd_detect_on_img(filename, save_img=False, saved_img_path=None):
    ssd_model = cv2.dnn.readNetFromCaffe(
    'ssd/deploy.prototxt',
    'ssd/res10_300x300_ssd_iter_140000.caffemodel')

    img = cv2.imread(filename)
    face_model = ssd_model
    face_blob_height = 300
    face_average_color = (104, 177, 123)
    face_confidence_threshold = 0.3

    while not img is None:
        h, w = img.shape[:2]
        aspect_ratio = w/h

        # Detect faces in the frame.

        face_blob_width = int(face_blob_height * aspect_ratio)
        face_blob_size = (face_blob_width, face_blob_height)

        face_blob = cv2.dnn.blobFromImage(
            img, size=face_blob_size, mean=face_average_color)

        face_model.setInput(face_blob)

        try:        
            face_results = face_model.forward()

            # Iterate over the detected faces.
            for face in face_results[0, 0]:
                face_confidence = face[2]
                if face_confidence > face_confidence_threshold:

                    # Get the face coordinates.
                    x0, y0, x1, y1 = (face[3:7] * [w, h, w, h]).astype(int)

                    # Draw a blue rectangle around the face.
                    cv2.rectangle(img, (x0, y0), (x1, y1),
                                    (255, 0, 0), 2)
            if save_img:
                cv2.imwrite(saved_img_path, img)
            cv2.imshow('Faces', img)

            if cv2.waitKey(1) == 27: # Escape
                cv2.destroyAllWindows()
                break
        except:
            print("No face detected on the image")
            break

def ssd_anonymize_on_video(filename, save_to_avi=False, show_fps=False):
    ssd_model = cv2.dnn.readNetFromCaffe(
    'ssd/deploy.prototxt',
    'ssd/res10_300x300_ssd_iter_140000.caffemodel')

    face_blob_height = 300
    face_average_color = (104, 177, 123)
    face_confidence_threshold = 0.9

    cap = cv2.VideoCapture(filename)

    success, frame = cap.read()

    if save_to_avi:
        
        size = (int(cap.get(3)), int(cap.get(4)))
        result = cv2.VideoWriter('results/anonymized_ssd.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
        frames = 0
        start_time = time.time()
        while success:
            h, w = frame.shape[:2]
            aspect_ratio = w/h

            face_blob_width = int(face_blob_height * aspect_ratio)
            face_blob_size = (face_blob_width, face_blob_height)

            face_blob = cv2.dnn.blobFromImage(
                frame, size=face_blob_size, mean=face_average_color)

            ssd_model.setInput(face_blob)
            face_results = ssd_model.forward()
            for face in face_results[0, 0]:
                face_confidence = face[2]
                if face_confidence > face_confidence_threshold:
                    x0, y0, x1, y1 = (face[3:7] * [w, h, w, h]).astype(int)
                    try:
                        frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (99, 99), 0)
                    except cv2.error:
                        pass
            frames += 1

            result.write(frame)
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
            h, w = frame.shape[:2]
            aspect_ratio = w/h

            face_blob_width = int(face_blob_height * aspect_ratio)
            face_blob_size = (face_blob_width, face_blob_height)

            face_blob = cv2.dnn.blobFromImage(
                frame, size=face_blob_size, mean=face_average_color)

            ssd_model.setInput(face_blob)
            face_results = ssd_model.forward()
            for face in face_results[0, 0]:
                face_confidence = face[2]
                if face_confidence > face_confidence_threshold:
                    x0, y0, x1, y1 = (face[3:7] * [w, h, w, h]).astype(int)
                    try:
                        frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (99, 99), 0)
                    except cv2.error:
                        pass
            frames += 1
            cv2.imshow('Faces', frame)
            if cv2.waitKey(1) == 27: # Escape
                cv2.destroyAllWindows()
                break

            success, frame = cap.read()
        end_time = time.time()
        duration = end_time - start_time
        fps = frames / duration
        if show_fps:
            print("FPS: {:.2f}".format(fps))