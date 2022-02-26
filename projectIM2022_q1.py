from ensurepip import version
import cv2 as cv
import numpy as np
import skimage
from matplotlib import pyplot as plt
#print(cv.__version__)
#print(np.__version__)
#print(skimage.__version__)

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
#cv.imshow("img1",img1)

#img2=cv.imread(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M42966-1-E.jpg")
#img2=resize_img(scale_percent=scale_percent,img=img2)
#cv.imshow("img2",img2)

#img3=cv.imread(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M43025-1-E.jpg")
#img3=resize_img(scale_percent=scale_percent,img=img3)
#cv.imshow("img3",img3)

#img4=cv.imread(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M43291-1-E.jpg")
#img4=resize_img(scale_percent=scale_percent,img=img4)
#cv.imshow("img4",img4)

edges = cv.Canny(img1,100,200)
plt.subplot(121),plt.imshow(img1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()