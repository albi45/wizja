import cv2
import time


def viola_detect_on_img(filename, save_img=False, saved_img_path=None):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.namedWindow('Face Detection')
    if save_img:
        cv2.imwrite(saved_img_path, img)
    cv2.imshow('Face Detection', img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


def viola_anonymize_on_video(filename, save_to_avi=False, show_fps=False):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(filename)
    success, frame = cap.read()

    if save_to_avi:
        
        size = (int(cap.get(3)), int(cap.get(4)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter('results/anonymized_viola.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, size)
        frames = 0
        start_time = time.time()
        while success:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                faces = frame[y:y + h, x:x + w]
                blurred = cv2.medianBlur(faces, 99)
                frame[y:y+h, x:x+w] = blurred
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                faces = frame[y:y + h, x:x + w]
                blurred = cv2.medianBlur(faces, 99)
                frame[y:y+h, x:x+w] = blurred
            frames += 1
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) == 27: # Escape
                cv2.destroyAllWindows()
                break
            success, frame = cap.read()
        end_time = time.time()
        duration = end_time - start_time
        fps = frames / duration
        if show_fps:
            print("FPS: {:.2f}".format(fps))

