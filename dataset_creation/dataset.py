import cv2
import os
import numpy as np
import shutil

from utils.common import resize
from consts.common import SEED


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


def get_negative(image, bbx, bbxs):
    np.random.seed(SEED)
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
    np.random.seed(SEED)
    directories = [f"../dataset/val/1", f"../dataset/val/0"]
    for directory in directories:
        folder = directory.split("/")[-1]
        files = list(os.listdir(directory))
        half = int(len(files)/2)
        np.random.shuffle(files)
        files = files[:half]
        for file in files:
            src = f"{directory}/{file}"
            shutil.move(src, f"../dataset/test/{folder}/{file}")


def main():
    dirs = ["../wider_face_split/wider_face_train_bbx_gt.txt", "../wider_face_split/wider_face_val_bbx_gt.txt"]
    prepare_dataset(dirs[0], "../WIDER_train/images/", "../dataset/train/")
    prepare_dataset(dirs[1], "../WIDER_val/images/", "../dataset/val/")


if __name__ == '__main__':
    main()