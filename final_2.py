import os
import cv2
import numpy as np

"""
    Authors: Lidor Eliyahu Shelef, Yuval Avni
"""


# noinspection PyUnresolvedReferences
def resize_img(_scale_percent, img):
    width = int(img.shape[1] * _scale_percent / 100)
    height = int(img.shape[0] * _scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    return resized


# showing the images with normal scale so my pc screen would be enough
# noinspection PyUnresolvedReferences
def find_stuff(image_name):
    image = cv2.imread(image_name)
    image = resize_img(scale_percent, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    # reading image
    img = image
    img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=3)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=3)
    # converting image into grayscale image
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
        # findContour function detects whole image as shape
        if i == 0:
            i = 1
            continue
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        # using drawContours() function
        cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)
        # noinspection PyUnreachableCode
        if False:
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
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


def final_2(image_full_location=""):
    # noinspection PyGlobalUndefined
    global scale_percent
    scale_percent = 10  # percent of original size
    if image_full_location == "":
        # Image Number 1
        image_1_name = f"{os.getcwd()}\\images\\M40967-1-E.jpg"
        find_stuff(image_1_name)
        # Image Number 2
        # image_2_name = f"{os.getcwd()}\\images\\M42966-1-E.jpg"
        # find_stuff(image_2_name)
        # Image Number 3
        # image_3_name = f"{os.getcwd()}\\images\\M43025-1-E.jpg"
        # find_stuff(image_3_name)
        # Image Number 4
        # image_4_name = f"{os.getcwd()}\\images\\M43291-1-E.jpg"
        # find_stuff(image_4_name)

    else:
        find_stuff(image_full_location)
    # noinspection PyUnresolvedReferences
    cv2.waitKey(0)
    # noinspection PyUnresolvedReferences
    cv2.destroyAllWindows()


final_2()
