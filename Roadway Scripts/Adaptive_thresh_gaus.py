import cv2 as cv
import numpy as np
from scipy import ndimage as ndi
def resize_img(scale_percent,img):
   
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)
    #cv.imshow("Resized image", resized)
    return resized


scale_percent = 10 # percent of original size

#showing the images with normal scale so my pc screen would be enogh

img1=cv.imread(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M40967-1-E.jpg")
img1=resize_img(scale_percent=scale_percent,img=img1)
cv.imshow("img1",img1)
gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

th=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
cv.imshow("th",th)

cv.waitKey(0)
cv.destroyAllWindows()