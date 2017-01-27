import sys
sys.path.append('/usr/local/Cellar/opencv3/3.0.0/lib/python2.7/site-packages')
import cv2
from random import randint
import numpy as np

# rgb(255, 241, 204)
# rgb(255, 213, 54)
# rgb(196, 157, 82)
# rgb(125, 66, 11)
# rgb(77, 41, 15)

def color(x, y, w, h, area):
    a = [
        [255, 255, 255],
        [255, 241, 204],
        [255, 213, 54],
        [196, 157, 82],
        [125, 66, 11],
                [255, 241, 204],
                [255, 213, 54],
                [196, 157, 82],
                [125, 66, 11],
        [77, 41, 15],
    ]
    if area > 20000: return a[0]
    # if area > 2000: return a[1]
    # if area > 1000: return a[2]
    # if area > 200: return a[3]
    # if area > 50: return a[4]
    # return a[5]

    return a[randint(1, 9)]
    # return [randint(0, 255), randint(0, 255), randint(0, 255)]

def test(path):
    img = cv2.imread(path)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY)
    output = cv2.connectedComponentsWithStats(thr, 4, cv2.CV_32S)
    # cv2.connectedComponents()
    num_labels = output[0]
    labels = output[1]
    labels_ = np.array(labels).astype(float) / num_labels * 255
    stats = output[2]
    centroids = output[3]
    print num_labels
    # print stats, centroids

    # imw, imh =
    imh = np.size(labels, 0)
    imw = np.size(labels, 1)

    dst = np.zeros((imh, imw, 3))

    colors = list()
    colors.append([255, 255, 255]) # rgb, int
    # colors.append([0, 0, 0]) # rgb, int
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # print area
        colors.append(color(x, y, w, h, area))

    for i in range(imh):
        for j in range(imw):
            # if labels[i, j] == 0: continue
            c = colors[labels[i, j]]
            dst[i, j, 0] = c[0]
            dst[i, j, 1] = c[1]
            dst[i, j, 2] = c[2]

    dst = cv2.erode(dst, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    for i in range(imh):
        for j in range(imw):
            img[i, j, 0] = min(img[i, j, 0], dst[i, j, 0])
            img[i, j, 1] = min(img[i, j, 1], dst[i, j, 1])
            img[i, j, 2] = min(img[i, j, 2], dst[i, j, 2])

    cv2.imwrite(path+'_labels.png', labels_)
    cv2.imwrite(path+'_thr.png', thr)
    cv2.imwrite(path+'_dst.png', dst)
    # cv2.imwrite('/Users/zhoubowei/Downloads/2__.jpg', grey - thr)

print cv2.__version__
test('/Users/zhoubowei/Downloads/3.png')
