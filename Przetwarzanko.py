import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randint


def remove_holes(img):
    im_in = img
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out


def draw(img):
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
#puchatek_puzzle_1
#cat_puzzle
#pawel1
#pawel2
#xd
chosen_img = 'pawel2.jpg'
img = cv2.imread(os.path.join('img', chosen_img))
#img2 = cv2.copyMakeBorder(img2, 50, 50, 50, 50, cv2.BORDER_REPLICATE, None)
draw(img)


def findObjects(img):
    blur = cv2.GaussianBlur(img, (7, 7), 2)
    h, w = img.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    lowerb = np.array([0, 0, 0])
    upperb = np.array([15, 15, 15])
    binary = cv2.inRange(gradient, lowerb, upperb)

    kern = np.ones((5, 5), np.uint8)

    test = remove_holes(binary)

    erosion = cv2.erode(test, kern, iterations=3)
    erosion = cv2.dilate(erosion, kern, iterations=1)
    binary = erosion

    tmp = cv2.imread(os.path.join('img', chosen_img))
    edged = cv2.Canny(binary, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    im2, cnts, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )

    for i in range(len(cnts)):
        if hierarchy[0, i, 3] == -1:
            cv2.drawContours(tmp, cnts, i, (255, 0, 255), 5, cv2.FILLED)

    print("Found %d objects." % len(cnts))

    mask = np.zeros(img.shape, dtype="uint8")
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return cnts

def add_padding(img, size=256):
    new_im = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None)
    return new_im


def findDistinctPuzzles(img, cnts):
    col = 0
    row = 0
    if len(cnts) % 2 == 0:
        col = 2
        row = int(len(cnts) / 2)
    else:
        col = 1
        row = len(cnts)
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, col * row + 1):
        fig.add_subplot(row, col, i)
        x, y, w, h = cv2.boundingRect(cnts[i - 1])
        if (w > 50 and h > 50):
            new_img = img[y:y + h, x:x + w]
        else:
            new_img = np.zeros([50, 50], dtype="uint8")
        plt.imshow(new_img)
        # draw(add_padding(new_img))
        draw(new_img)

    plt.show()


