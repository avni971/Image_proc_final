from time import sleep
from matplotlib import pyplot as plt
import cv2
from cv2 import cvtColor
import numpy as np

#image = cv2.imread('M40967-1-E.jpg')

def resize_img(scale_percent,img):
   
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)
    #cv.imshow("Resized image", resized)
    return resized


scale_percent = 10 # percent of original size

#showing the images with normal scale so my pc screen would be enogh
def find_stuff(imagename):
    image=cv2.imread(imagename)
    image=resize_img(scale_percent,image)

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)

    
    # reading image
    img = image
    img=cv2.erode(img,np.ones((3,3),np.uint8),iterations=3)
    img=cv2.dilate(img,np.ones((3,3),np.uint8),iterations=3)
    # converting image into grayscale image
    #cv2.imshow("after erode,dialte",img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    
    # list for storing names of shapes
    for contour in contours:
    
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
    
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # using drawContours() function
        cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)
        if False:
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
        
            # putting shape name at center of each shape
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                cv2.putText(img, 'Quadrilateral', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                cv2.putText(img, 'Pentagon', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                cv2.putText(img, 'Hexagon', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                cv2.putText(img, 'circle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # displaying the image after drawing contours
    cv2.imshow('shapes', image)


#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M40967-1-E.jpg")

#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M42966-1-E.jpg")

#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M43025-1-E.jpg")

#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M43291-1-E.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()