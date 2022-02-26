import sys
import cv2
import numpy
import random
from scipy.ndimage import label
def resize_img(scale_percent,img):
   
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)
    #cv.imshow("Resized image", resized)
    return resized



def segment_on_dt(img):
    dt = cv2.distanceTransform(img, 2, 3) # L2 norm, 3x3 mask
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)[1]
    lbl, ncc = label(dt)

    lbl[img == 0] = lbl.max() + 1
    lbl = lbl.astype(numpy.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lbl)
    lbl[lbl == -1] = 0
    return lbl


scale_percent=10
img1=cv2.imread(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M42966-1-E.jpg")
img1=resize_img(scale_percent=scale_percent,img=img1)
cv2.imshow("img1",img1)


gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)

img = gray
img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
img = 255 - img # White: objects; Black: background

ws_result = segment_on_dt(img)
# Colorize
height, width = ws_result.shape
ws_color = numpy.zeros((height, width, 3), dtype=numpy.uint8)
lbl, ncc = label(ws_result)

for l in range(1, ncc + 1):
    a, b = numpy.nonzero(lbl == l)
    if img[a[0], b[0]] == 0: # Do not color background.
        continue
    rgb = [random.randint(0, 255) for _ in range(3)]
    ws_color[lbl == l] = tuple(rgb)

cv2.imwrite("sys.argv[2].png", ws_color)

cv2.waitKey(0)
cv2.destroyAllWindows()