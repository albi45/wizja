from detection.detection import anonymize_on_video, cascade_detect_video
from existing.viola.viola import viola_anonymize_on_video
from existing.mtcnn.mtcnn import mtcnn_anonymize_on_video
from existing.ssd.ssd import ssd_anonymize_on_video

filename = "the_office.mp4"

def main():
    # anonymize_on_video(filename=filename, use_webcam=False, save_to_avi=True)
    anonymize_on_video()
    # viola_anonymize_on_video(filename=filename, save_to_avi=True)
    # mtcnn_anonymize_on_video(filename=filename, save_to_avi=True)
    # ssd_anonymize_on_video(filename=filename, save_to_avi=True)
    # cascade_detect_video(show_fps=True)
    # viola_anonymize_on_video(0, show_fps=True)
    # mtcnn_anonymize_on_video(0, show_fps=True)
    # ssd_anonymize_on_video(0, show_fps=True)


if __name__ == "__main__":
    main()
