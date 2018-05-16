import cv2
import numpy as np

from sliding.sliding_windows import pyramid
from sliding.sliding_windows import sliding_window
from sliding.sliding_windows import non_max_suppression_fast as nms

from sliding.detector_up import pen_detector
from sliding.detector_up import bow_features

def in_range(number, test, thresh=0.2):
    return abs(number - test) < thresh

def detector(img):

    svm, extractor = pen_detector()
    detect = cv2.xfeatures2d.SIFT_create()

    w, h = 100, 230

    rectangles = []
    counter = 1
    scaleFactor = 1.5
    scale = 1
    font = cv2.FONT_HERSHEY_PLAIN

    for resized in pyramid(img, scaleFactor):
        scale = float(img.shape[1]) / float(resized.shape[1])
        for (x, y, roi) in sliding_window(resized, 20, (w, h)):
            if roi.shape[1] != w or roi.shape[0] != h:
                continue

            try:
                bf = bow_features(roi, extractor, detect)
                _, result = svm.predict(bf)
                a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                print("class: %d, score: %f" % (result[0][0], res[0][0]))
                score = res[0][0]
                if result[0][0] == 1:
                    if score < -1.4:
                        rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)
                        rectangles.append([rx, ry, rx2, ry2, abs(score)])
            except:
                pass

            counter += 1

    windows = np.array(rectangles)
    boxes = nms(windows, 0.1)
    boxes2 = boxes[:, :4]
    print(boxes2)
    return(boxes2)
