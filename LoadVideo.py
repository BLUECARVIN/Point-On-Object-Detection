import cv2
from pylab import *

def LoadVideo(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("the total of frames is %d" % count)
    while True:
        rt, im = cap.read()
        frames.append(im)
        count -= 1
        if count == 0:
            break
    frames = array(frames)
    return frames