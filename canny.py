import cv2 as cv
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data,img_as_float
import matplotlib.pyplot as plt
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

print(img1.shape)

gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

img_edge=cv.Canny(img1,100,150, L2gradient = True)
cv.imshow("image_edge",img_edge)

#result= cv.cvtColor(img_edge,cv.COLOR_GRAY2RGB)
#cv.rectangle(result,(100,150),(150,200),(0,0,255),2)
#cv.imshow("red",result)


cv.waitKey(0)
cv.destroyAllWindows()