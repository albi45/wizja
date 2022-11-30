import cv2


def read_data(path, data_path):
    names = []
    labels = []
    images = []
    next_bbs = False
    number_to_read = 0
    with open(path, "r") as f:
        for line in f:
            if ".jpg" in line:
                names.append(line.rstrip())
                next_bbs = True
            elif next_bbs:
                number_to_read = int(line.rstrip())
                next_bbs = False
            elif number_to_read > 0:
                values = line.split(" ")
                labels.append((values[0], values[1], values[2], values[3]))
                number_to_read -= 1
    for name in names:
        images.append(cv2.imread(data_path + name))
    return names, labels


dirs = ["wider_face_split/wider_face_train_bbx_gt.txt", "wider_face_split/wider_face_val_bbx_gt.txt"]
train_names, train_labels = read_data(dirs[0], "WIDER_train/images/")
val_names, val_labels = read_data(dirs[1], "WIDER_val/images/")
