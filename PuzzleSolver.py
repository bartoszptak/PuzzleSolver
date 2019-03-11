import cv2
import numpy as np
import matplotlib.pyplot as plt


class PuzzleSolver:
    def __init__(self):
        pass

    def draw(self, img):
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    def get_binary_image_from_bgr(self,img):
        loc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        loc = cv2.medianBlur(loc, ksize=3)
        loc = cv2.threshold(loc, 60, 255, cv2.THRESH_BINARY)[1]
        loc = cv2.dilate(loc, (5,5))
        loc = cv2.medianBlur(loc, ksize=9)
        return cv2.blur(loc, ksize=(3, 3))

    def get_borders_from_binary(self, binary_image):
        dst = cv2.cornerHarris(np.float32(binary_image),2,3,0.04)
        dst = cv2.dilate(dst,None)
        
        im = np.zeros((dst.shape[0], dst.shape[1]), dtype=np.uint8)
        im[dst<0.03*np.min(dst)]=[255]
        im[dst>0.03*np.max(dst)]=[255]
        return im

    def calculate_lines_moments_from_borders(self, border_image):
        lines = cv2.HoughLines(border_image, 0.1, np.pi/180, 100)
        
        li = []
        for el in lines:
            li.append(el[0])

        wyn = [[li[0]]]    

        res = [li[0]]

        for i in range(1, len(li)):
            flag = False
            for k in range(len(res)):
                if (res[k][0]-5 < li[i][0] and li[i][0] < res[k][0]+5) and (res[k][1]-5 < li[i][1] and li[i][1] < res[k][1]+5):
                    wyn[k].append(li[i])
                    flag = True
            if not flag:
                wyn.append([li[i]])
                res.append(li[i])

        moments = []

        for el in wyn:
            a, b, c = 0, 0, 0
            for e in el:
                a += e[0]
                b += e[1]
                c += 1
            moments.append([a/c, b/c])
            
        return moments

    def from_moments_calculate_border_points(self, moments):
        points = []

        for el in moments:
            rho,theta = el[0], el[1]
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a*rho
            y0 = b*rho

            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            points.append([[x1,y1],[x2,y2]])
        
        return points

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return y, x

    def from_border_points_calculate_corners(self, border_points):
        corners = set()
        for i in range(len(border_points)):
            for k in range(len(border_points)):
                lx = self.line_intersection(border_points[i], border_points[k])
                if lx is not None:
                    corners.add(lx)

        corners = np.array(list(corners), dtype=np.int32)
        return sorted(corners, key=lambda x: (x[0], x[1]), reverse=False)

    def from_corners_and_borders_calculate_sides(self, corners, border_image):
        lu = corners[0]
        pu = corners[1]
        ld = corners[2]
        pd = corners[3]

        cx, cy = np.int32(self.line_intersection([lu,pd],[pu,ld]))

        up, down, left, right = [], [], [], []
        padl, padr, padu, padd = 0, 0, 0, 0 

        for i in range(0, cy):
            if border_image[0, cx] != border_image[i, cx]:
                padu = i
                break

        for i in range(border_image.shape[0]-1, cy, -1):
            if border_image[border_image.shape[0]-1, cx] != border_image[i, cx]:
                padd = i
                break
            
        for i in range(0, cx):
            if border_image[cy, 0] != border_image[cy, i]:
                padl = i
                break
            
        for i in range(border_image.shape[1]-1, cx, -1):
            if border_image[cy, border_image.shape[1]-1] != border_image[cy, i]:
                padr = i
                break
            
        for i in range(border_image.shape[0]):
            for k in range(border_image.shape[1]):
                if border_image[i,k] != 0:
                    if k >= lu[1]+5 and k <= pu[1]-5 and (i <= padu+5 or i <= np.mean([lu[0],pu[0]])+5):
                        up.append([i,k])
                    if k > ld[1]+5 and k < pd[1]-5 and (i >= padd-5 or i>=np.mean([ld[0],pd[0]])-5):
                        down.append([i,k])
                    if i > lu[0]+5 and i < ld[0]-5 and (k <= padl+5 or k<=np.mean([lu[1],ld[1]])+5):
                        left.append([i,k])
                    if i > pu[0]+5 and i < pd[0]-5 and (k >= padr-5 or k>=np.mean([pu[0],pd[0]])-5):
                        right.append([i,k])
        return up, down, left, right

    def check_slides(self, up, down, left, right):
        if np.std(np.array(up)[:,0]) < 5.0:
            print('góra')
        if np.std(np.array(down)[:,0]) < 5.0:
            print('dół')
        if np.std(np.array(left)[:,1]) < 5.0:
            print('lewo')
        if np.std(np.array(right)[:,1]) < 5.0:
            print('prawo')

    def find_contours(self, image):
        binary = self.get_binary_image_from_bgr(puzz)
        border_image = self.get_borders_from_binary(binary)
        moments = self.calculate_lines_moments_from_borders(border_image)
        border_points = self.from_moments_calculate_border_points(moments)
        corners = self.from_border_points_calculate_corners(border_points)
        up_side, down_side, left_side, right_side = self.from_corners_and_borders_calculate_sides(corners, border_image)
        print(self.check_slides(up_side, down_side, left_side, right_side))

    ### MARIAN I PAWEŁ ZONE

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

        print("Found %d objects." % len(cnts))

        return cnts

    def add_padding(self, img, size=256):
        new_im = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None)
        return new_im

    def find_distinct_puzzles(self, img, cnts):
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
            self.draw(new_img)

        plt.show()

    def split_puzzles_from_image(self, img):
        cnts = self.find_objects(img)
        self.find_distinct_puzzles(img, cnts)

if __name__ == '__main__':
    ps = PuzzleSolver()

    img = cv2.imread('img/pawel1.jpg')
    
    #ps.split_puzzles_from_image(img)

    #puzz = img[0:285, 650:900]
    #ps.find_contours(puzz)
