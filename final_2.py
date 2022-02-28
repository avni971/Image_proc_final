import csv
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
    return resized


# showing the images with normal scale so my pc screen would be enough
# noinspection PyUnresolvedReferences
def find_stuff(image_name):
    image = cv2.imread(image_name)
    image = resize_img(scale_percent, image)
    cv2.imshow("image", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    head, tail = os.path.split(image_name)
    tail = os.getcwd() + "\\Exports\\" + tail[:len(tail) - 4]
    tail += "-shapes.csv"
    cv2.imshow("shapes", image)
    with open(tail, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(contours)

    # noinspection PyUnresolvedReferences
    cv2.waitKey(0)
    # noinspection PyUnresolvedReferences
    cv2.destroyAllWindows()


def absolute_file_paths(directory):
    for dir_path, _, filenames in os.walk(directory):
        for file_ref in filenames:
            yield os.path.abspath(os.path.join(dir_path, file_ref))


def final__2(image_full_location: str = None, folder_location: str = None):
    """
        Only one of the variables should and can be assigned at a time
        :param image_full_location: Flag indicates that the desire action is to run this function on a single image
        :type image_full_location: str
        :example image_full_location: for example C:users/Lidor/ImageProcessing/FinalProject/images/image_1.jpg
        :param folder_location: Flag that indicates we should run on the entire folder that is given
        :type folder_location: str
        :example folder_location: for example C:users/Lidor/ImageProcessing/FinalProject/images/
    """
    image_formats = [
        'png', 'jpg', 'jpeg'
    ]
    # noinspection PyGlobalUndefined
    global scale_percent
    scale_percent = 10  # percent of original size
    if image_full_location is not None and folder_location is None:
        # image_1_name = f"{os.getcwd()}\\images\\M40967-1-E.jpg"
        find_stuff(image_full_location)
    elif folder_location is not None and image_full_location is None:
        images_in_folder = absolute_file_paths(folder_location)  # os.listdir(folder_location)
        for idx, image_i in enumerate(images_in_folder):
            if image_i.split('.')[-1] in image_formats:
                print(f"Opening image {idx}")
                find_stuff(image_i)
                print(f"Closing image {idx}")
    else:
        print("Please make sure that at least one variable is defined (and only one of them)")


# final__2(image_full_location="D:/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/M40967-1-E.jpg")
# final__2(folder_location="D:/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/")

# python -c "import final_2; from final_2 import final__2; final__2(
# image_full_location='D:/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/M40967-1-E.jpg')"
# python -c "import final_2; from final_2 import final__2; final__2(
# folder_location='D:/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/')"
