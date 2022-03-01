import csv
import math
import cv2
import os
import numpy as np

"""
    Authors: Yuval Avni, Lidor Eliyahu Shelef
"""


def resize_img(_scale_percent, img):
    width = int(img.shape[1] * _scale_percent / 100)
    height = int(img.shape[0] * _scale_percent / 100)
    dim = (width, height)
    # noinspection PyUnresolvedReferences
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def contain(point1, point2, delta):
    x1 = point1[0]
    y1 = point1[1]
    w1 = point1[2]
    h1 = point1[3]
    x2 = point2[0]
    y2 = point2[1]
    w2 = point2[2]
    h2 = point2[3]
    if x1 < x2 - delta and y1 < y2 - delta and x2 + w2 + delta < x1 + w1 and y2 + h2 + delta < y1 + h1:
        return 1
    else:
        return 0


def contain_2(point1, point2, delta):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point1[2]
    y2 = point1[3]
    x3 = point2[0]
    y3 = point2[1]
    x4 = point2[2]
    y4 = point2[3]
    if x1 < x3 - delta and y1 < y3 - delta and x4 + delta < x2 and y4 + delta < y2:
        return 1
    else:
        return 0


def overlap(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point1[2]
    y2 = point1[3]
    x3 = point2[0]
    y3 = point2[1]
    x4 = point2[2]
    y4 = point2[3]
    # Calculations are as follows:
    #  (right top overlap)   or    (right bottom overlap)  or    (left bottom overlap)    or   (left top overlap)
    if (x1 < x3 < x2 and y1 < y4 < y2) or (x1 < x3 < x2 and y1 < y3 < y2) or (x1 < x4 < x2 and y1 < y3 < y2) or (
            x1 < x4 < x2 and y1 < y4 < y2):
        return True
    else:
        return False


def sort_by_distance(valid_contour):
    distance_array = []
    for row in valid_contour:
        distance_array.append((round(math.dist((row[0], row[1]), (0, 0)), 2), row))
    sorted_array = sorted(distance_array, key=lambda x: (x[0]))

    sorted_points = []
    for row in sorted_array:
        sorted_points.append(row[1])
    return sorted_points


# noinspection PyUnresolvedReferences
def write_number(img, number, x1, y1, x2, y2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (((x1 + x2) // 2) - 2, ((y1 + y2) // 2) + 2)
    font_Scale = 0.3
    font_Color = (255, 0, 0)
    thickness = 1
    line_Type = 1

    cv2.putText(img, str(number),
                bottomLeftCornerOfText,
                font,
                font_Scale,
                font_Color,
                thickness,
                line_Type)
    return img


def close_contour(R1, R2, delta):
    if (R1[0] > R2[2] + delta) or (R1[2] < R2[0] - delta) or (R1[3] < R2[1]) or (R1[1] > R2[3]):
        return False
    else:
        return True


# noinspection PyUnresolvedReferences
def add_number(_sorted_valid_contour, _image_copy, _image_copy_copy, image_name):
    alpha = 0
    for i, row1 in enumerate(_sorted_valid_contour):
        for j, row2 in enumerate(_sorted_valid_contour):
            if row1 != row2:
                if abs(row1[0] - row2[0]) + abs(row1[1] - row2[1]) + abs(row1[2] - row2[2]) + abs(
                        row1[3] - row2[3]) <= alpha:
                    row2[0] = min(row1[0], row2[0])
                    row2[1] = min(row1[1], row2[1])
                    row2[2] = max(row1[2], row2[2])
                    row2[3] = max(row1[3], row2[3])

                    _sorted_valid_contour = _sorted_valid_contour[:j] + _sorted_valid_contour[j + 1:]

    for i, row in enumerate(_sorted_valid_contour):
        _image_copy_copy = write_number(_image_copy_copy, i, row[0], row[1], row[2], row[3])
        _image_copy = write_number(_image_copy, i, row[0], row[1], row[2], row[3])
        cv2.rectangle(_image_copy, (row[0], row[1]), (row[2], row[3]), (0, 0, 255), 2)
        cv2.rectangle(_image_copy_copy, (row[0], row[1]), (row[2], row[3]), (0, 0, 255), 2)

    cv2.imshow("boxes", _image_copy)
    head, tail = os.path.split(image_name)
    tail = os.getcwd() + "\\Exports\\" + tail[:len(tail) - 4]
    tail += "-box.csv"
    with open(tail, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(["left_up_x", "left_up_y", "left_down_x", "left_down_y"])
        write.writerows(_sorted_valid_contour)


# showing the images with normal scale so my pc screen would be enough
# noinspection PyUnresolvedReferences
def apply_box_detection(image_name, delta=10):
    image = cv2.imread(image_name)
    image = resize_img(scale_percent, image)
    cv2.imshow("image", image)

    image_copy = image
    image_copy_copy = np.zeros(image.shape, np.uint8)
    _, image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    point_array = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        point_array.append([x, y, w, h])

    valid_contour = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        w = w + 10
        h = h + 10
        if 400 > math.sqrt(math.pow(h, 2) + math.pow(w, 2)) > 5 and delta < x < image_copy.shape[1] - delta \
                and delta < y < image_copy.shape[0] - delta:
            count = 0
            for point1 in point_array:
                # if point2 in point1
                count += contain(point1, [x, y, w, h], delta=4)
            if count <= 2:
                # cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,0,255),2)
                # cv2.rectangle(image_copy_copy,(x,y),(x+w,y+h),(0,0,255),2)
                valid_contour.append([x, y, x + w, y + h])

    valid_contour_temp = []
    if True:
        for i, row1 in enumerate(valid_contour):
            temp_row = row1
            for j, row2 in enumerate(valid_contour):
                if (math.dist((row2[0], row2[1]), (row2[2], row2[3]))) < 35 and i != j:
                    close = overlap(row1, row2)
                else:
                    close = False
                if close:
                    # print("y")
                    # print(i, j)
                    # print(row1, row2)
                    temp_row[0] = min(row1[0], row2[0])
                    temp_row[1] = min(row1[1], row2[1])
                    temp_row[2] = max(row1[2], row2[2])
                    temp_row[3] = max(row1[3], row2[3])
            valid_contour_temp.append(temp_row)

    sorted_valid_contour = sort_by_distance(valid_contour_temp)

    non_valid_contour_temp2 = []
    for i, row1 in enumerate(valid_contour_temp):
        for j, row2 in enumerate(valid_contour_temp):
            if contain_2(row1, row2, delta=0) == 1:
                non_valid_contour_temp2.append(row2)

    for element in valid_contour_temp:
        if element in non_valid_contour_temp2:
            valid_contour_temp.remove(element)

    sorted_valid_contour_2 = sort_by_distance(valid_contour_temp)

    # add_number(sorted_valid_contour,image_copy,image_copy_copy)
    add_number(sorted_valid_contour_2, image_copy, image_copy_copy, image_name)
    # noinspection PyUnresolvedReferences
    cv2.waitKey(0)
    # noinspection PyUnresolvedReferences
    cv2.destroyAllWindows()


def absolute_file_paths(directory):
    for dir_path, _, filenames in os.walk(directory):
        for file_ref in filenames:
            yield os.path.abspath(os.path.join(dir_path, file_ref))


def apply_ex1_detection(image_full_location: str = None, folder_location: str = None):
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
        apply_box_detection(image_full_location, delta=20)
    elif folder_location is not None and image_full_location is None:
        images_in_folder = absolute_file_paths(folder_location)  # os.listdir(folder_location)
        for idx, image_i in enumerate(images_in_folder):
            if image_i.split('.')[-1] in image_formats:
                print(f"Opening image {idx}")
                apply_box_detection(image_i, delta=20)
                print(f"Closing image {idx}")
    elif folder_location is None and image_full_location is None:
        images_in_folder = absolute_file_paths(os.getcwd() + "/images/")  # os.listdir(folder_location)
        for idx, image_i in enumerate(images_in_folder):
            if image_i.split('.')[-1] in image_formats:
                print(f"Opening image {idx}")
                apply_box_detection(image_i)
                print(f"Closing image {idx}")
    else:
        print("Please make sure that ONLY one variable is defined!")


# Driver
# apply_ex1_detection(image_full_location="D:/Lidor/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/M40967-1-E.jpg")
# apply_ex1_detection(folder_location="D:/Lidor/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/")
# apply_ex1_detection()

# python -c "import projectIM2022_q1; from projectIM2022_q1 import apply_ex1_detection; apply_ex1_detection(
# image_full_location='D:/Lidor/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/M40967-1-E.jpg')"
# python -c "import projectIM2022_q1; from projectIM2022_q1 import apply_ex1_detection; apply_ex1_detection(
# folder_location='D:/Lidor/Study/P.Languages/Python/Workspace_2019/ImageProcessing/Final/images/')"
