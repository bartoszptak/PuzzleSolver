import cv2
import numpy as np
import matplotlib.pyplot as plt


class PuzzleSolver:
    def __init__(self):
        pass

    #region STATIC METHODS
    @staticmethod
    def draw(img):
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        plt.show()

    @staticmethod
    def add_padding(img, size=50, color=(255,255,255)):
        new_im = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=color)
        return new_im

    @staticmethod
    def get_intersect(a1, a2, b1, b2):
        s = np.vstack([a1,a2,b1,b2])
        h = np.hstack((s, np.ones((4, 1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z == 0:
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    #endregion

    #region SPLIT PUZZLES

    def remove_holes(self, img):
        im_in = img
        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()

        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = im_th | im_floodfill_inv
        return im_out

    def find_objects(self, img):
        blur = cv2.GaussianBlur(img, (7, 7), 2)
        h, w = img.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        lowerb = np.array([0, 0, 0])
        upperb = np.array([15, 15, 15])
        binary = cv2.inRange(gradient, lowerb, upperb)

        kern = np.ones((5, 5), np.uint8)

        test = self.remove_holes(binary)

        erosion = cv2.erode(test, kern, iterations=3)
        erosion = cv2.dilate(erosion, kern, iterations=1)
        binary = erosion

        edged = cv2.Canny(binary, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        im2, cnts, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )

        return cnts

    def find_distinct_puzzles(self, img, cnts):
        col = 0
        row = 0
        padd = 20

        if len(cnts) % 2 == 0:
            col = 2
            row = int(len(cnts) / 2)
        else:
            col = 1
            row = len(cnts)

        images = []
        for i in range(1, col * row + 1):
            x, y, w, h = cv2.boundingRect(cnts[i - 1])
            if (w > 50 and h > 50):
                images.append(img[y-padd:y + h+padd, x-padd:x + w+padd])

        return images

    def split_puzzles_from_image(self, img):
        cnts = self.find_objects(img)
        return self.find_distinct_puzzles(img, cnts)

    #endregion


    #region TRANSFORM PUZZLES

    def get_binary_image_from_bgr(self, img):
        blur = cv2.GaussianBlur(img, (7, 7), 2)
        h, w = img.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        lowerb = np.array([0, 0, 0])
        upperb = np.array([15, 15, 15])
        binary =  cv2.inRange(gradient, lowerb, upperb)
        binary = self.add_padding(binary)

        kern = np.ones((5, 5), np.uint8)
        test = self.remove_holes(binary)

        erosion = cv2.erode(test, kern, iterations=3)
        return cv2.dilate(erosion, kern, iterations=1)

    def get_borders_from_binary(self, binary_image):
        dst = cv2.cornerHarris(np.float32(binary_image),2,3,0.04)

        im = np.zeros((dst.shape[0], dst.shape[1]), dtype=np.uint8)
        im[dst!=0]=[255]
        return cv2.medianBlur(im, 5)
    
    def get_HoughLines_from_binary(self, a):
        HoughLines = cv2.HoughLines(a, 1, np.pi / 180, 105)

        lines = []
        for line in HoughLines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 500*(-b))
                y1 = int(y0 + 500*(a))
                x2 = int(x0 - 500*(-b))
                y2 = int(y0 - 500*(a))

                lines.append([x1,y1,x2,y2])                
        return lines

    def check_quadrant(self, el, shape, reku=10):
        x1, y1, x2, y2 = el[0], el[1], el[2], el[3]
        h, w = shape[0]/2, shape[1]/2
        
        if (x1>w and y1<h and x2>w and y2>h) or (x2>w and y2<h and x1>w and y1>h):
            return 0
        elif (x1<w and y1<h and x2<w and y2>h) or (x2<w and y2<h and x1<w and y1>h):
            return 1
        elif (x1<w and y1<h and x2>w and y2<h) or (x2<w and y2<h and x1>w and y1<h):
            return 2
        elif (x1<w and y1>h and x2>w and y2>h) or (x2<w and y2>h and x1>w and y1>h):
            return 3
        else:
            if y1<0 and x1>0 and x2>0 and y2>0:
                return 0
            return 4

    def get_centroids_from_lines(self, lines, shape):
        kmeans = []
        for el in lines:
            kmeans.append(self.check_quadrant(el, shape))
        
        dictionary = dict()
        dictionary[0], dictionary[1], dictionary[2], dictionary[3], dictionary[4] = list(), list(), list(), list(), list()

        for index, row in enumerate(np.array(lines)):
            dictionary[kmeans[index]].append(row)
            
        for el in dictionary[0]:
            if el[1] < 0:
                el[0], el[2] = el[2], el[0]
                el[1], el[3] = el[3], el[1]
        
        for el in dictionary[1]:
            if el[1] > 0:
                el[0], el[2] = el[2], el[0]
                el[1], el[3] = el[3], el[1]
        
        for el in dictionary[2]:
            if el[0] < 0:
                el[0], el[2] = el[2], el[0]
                el[1], el[3] = el[3], el[1]
                
        for el in dictionary[3]:
            if el[0] > 0:
                el[0], el[2] = el[2], el[0]
                el[1], el[3] = el[3], el[1]
                
        return [np.nanmean(dictionary[0], axis=0).astype('int'),np.nanmean(dictionary[1], axis=0).astype('int'),
            np.nanmean(dictionary[2], axis=0).astype('int'),np.nanmean(dictionary[3], axis=0).astype('int')]

    def from_centroids_get_lines(self, c, im):
        points = []
        for i in range(4):
            for k in range(i, 4):
                inter = self.get_intersect((c[i][0], c[i][1]), (c[i][2], c[i][3]), (c[k][0], c[k][1]), (c[k][2], c[k][3]))
                if not np.isinf(inter[0]) and im.shape[1] > inter[0] > 0 and im.shape[0] > inter[1] > 0:
                    points.append(inter)

        return np.array(points)

    def four_point_transform(self, im, pts_src):
        pts_dst = np.array([[200,200],[800, 200],[ 200,800],[800, 800]])

        im_dst = np.full((1000, 1000, 3), 0, np.uint8)

        h, status = cv2.findHomography(pts_src, pts_dst)

        return cv2.warpPerspective(im, h, (im_dst.shape[1],im_dst.shape[0]), borderValue=(0,0,0))

    #endregion

    def get_transform_image(self, img):
        binary = self.get_binary_image_from_bgr(img)
        im = np.copy(self.add_padding(img))
        borders = self.get_borders_from_binary(binary)
        lines = self.get_HoughLines_from_binary(borders)
        c = self.get_centroids_from_lines(lines, im.shape)
        points = self.from_centroids_get_lines(c, im)
    
        return self.four_point_transform(im, points), self.four_point_transform(borders, points)

if __name__ == '__main__':
    ps = PuzzleSolver()

    img = cv2.imread('img/pawel1.jpg')
    
    #ps.split_puzzles_from_image(img)

    #puzz = img[0:285, 650:900]
    #ps.find_contours(puzz)
