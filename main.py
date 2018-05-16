import lktrack
from pylab import *
import LoadVideo
from main_detector import detector

filename = 'testpen.MP4'
imnames = LoadVideo.LoadVideo(filename)
print(imnames.shape)
img = imnames[0]
box = detector(img)
lkt = lktrack.LKTracker(imnames, box)

for im, ft in lkt.track():
    print ('tracking %d features' % len(ft))

figure()
imshow(im)
for p in ft:
    plot(p[0], p[1], 'bo')
for t in lkt.tracks:
    plot([p[0] for p in t], [p[1] for p in t])
axis('off')
show()
