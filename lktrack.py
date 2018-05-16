from numpy import *
import cv2


lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

subpix_params = dict(zeroZone=(-1, -1), winSize=(10,10),
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

feature_params = dict(maxCorners=500, qualityLevel=0.01
                      , minDistance=100)


class LKTracker(object):

    def __init__(self, imnames, box):

        self.imnames = imnames
        self.features = []
        self.tracks = []
        self.current_frame = 0
        self.box = box
        self.frames_count = self.imnames.shape[0]


    def step(self, framenbr=None):

        if framenbr is None:
            self.current_frame = (self.current_frame + 1) % self.frames_count
        else:
            self.current_frame = framenbr % self.frames_count

    def detect_points(self):

        self.image = self.imnames[self.current_frame]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        features = cv2.goodFeaturesToTrack(self.gray, **feature_params)

        cv2.cornerSubPix(self.gray,features, **subpix_params)
        true_features = []
        num = features.shape[0]
        for (x1, y1, x2, y2) in self.box:
            # x1, x2 = 1280, 1380
            # y1, y2 = 420, 650
            print(num)
            for i in range(num -1):
                if features[i][0][0] > x1 and features[i][0][0] < x2 and features[i][0][1] > y1 and features[i][0][1] < y2:
                    true_features.append(features[i])
        features = true_features
        features = array(features)
        self.features = features
        self.tracks = [[p] for p in features.reshape((-1,2))]

        self.prev_gray = self.gray

    def track_points(self):

        if self.features != []:
            self.step()

            self.image = self.imnames[self.current_frame]
            self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

            tmp = float32(self.features).reshape(-1, 1, 2)

            features,status,track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray,self.gray,tmp,None,**lk_params)

            self.features = [p for (st,p) in zip(status,features) if st]

            features = array(features).reshape((-1,2))
            for i,f in enumerate(features):
                self.tracks[i].append(f)
            ndx = [i for (i,st) in enumerate(status) if not st]
            ndx.reverse()
            for i in ndx:
                self.tracks.pop(i)

            self.prev_gray = self.gray

    def track(self):

        for i in range(self.frames_count):
            if self.features == []:
                self.detect_points()
            else:
                self.track_points()

            f = array(self.features).reshape(-1,2)
            im = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
            yield im,f

    def draw(self):

        for point in self.features:
            cv2.circle(self.image,(int(point[0][0]),int(point[0][1])),3,(0,255,0),-1)

        cv2.imshow('LKtrack', self.image)
        cv2.waitKey(0)
