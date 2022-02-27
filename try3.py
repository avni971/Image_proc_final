import cv2
import numpy as np
import os


# noinspection PyUnresolvedReferences
def resize_img(scale_percent, img):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    # cv2.imshow("Resized image", resized)
    return resized


scale_percent = 10  # percent of original size


# showing the images with normal scale so my pc screen would be enough
# noinspection PyUnresolvedReferences
def find_stuff(imagename, iter):
    image = cv2.imread(imagename)
    image = resize_img(scale_percent, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("thresh",thresh)
    if True:
        kernel = np.ones((3, 3), np.uint8)
        work = thresh
        for x in range(iter):
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("opening",work)
        for x in range(iter):
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("closing",work)

        # cv2.imshow("work",work)
        gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("gradient", gradient)
        gradient_2 = cv2.morphologyEx(work, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("gradient_2", gradient_2)

        img_edge = cv2.Canny(gradient_2, 100, 150, L2gradient=True)
        cv2.imshow("image_edge", img_edge)

        # houg transorm
        lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 200)
        for i in range(0, len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("image", image)


image_1_name = f"{os.getcwd()}\images\M40967-1-E.jpg"
image_2_name = f"{os.getcwd()}\images\M42966-1-E.jpg"
image_3_name = f"{os.getcwd()}\images\M43025-1-E.jpg"
image_4_name = f"{os.getcwd()}\images\M43291-1-E.jpg"

find_stuff(image_1_name, 2)
# find_stuff(image_2_name, 3)
# find_stuff(image_3_name, 3)
# find_stuff(image_4_name, 3)

# noinspection PyUnresolvedReferences
cv2.waitKey(0)
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows()
