# This file will contain all the methods related to the input streams for inference.
import cv2


def video_to_frame(path):
    list_frame = []
    cap = cv2.VideoCapture(path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        list_frame.append(frame)

    return list_frame