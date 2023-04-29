import cv2
import numpy as np
import colorsys


class MouseEvent:
    def __init__(self, window_name, img):
        self.window_name = window_name
        cv2.setMouseCallback(window_name, self.mouseRGB, img)
        self.colorsB = 0
        self.colorsG = 0
        self.colorsR = 0
        self.RGB = np.zeros(3)
        self.click_num = 0

    def mouseRGB(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
            self.click_num += 1
            self.x = x
            self.y = y
            self.colorsB = self.RGB[2] = param[y, x, 2]
            self.colorsG = self.RGB[1] = param[y, x, 1]
            self.colorsR = self.RGB[0] = param[y, x, 0]

            self.HSV = colorsys.rgb_to_hsv(
                self.colorsR, self.colorsG, self.colorsB)

            # print("Red: ",self.colorsR)
            # print("Green: ",self.colorsG)
            # print("Blue: ",self.colorsB)
            # print("RGB Format: ",self.RGB)
            # print("Coordinates of pixel: X: ",self.x,"Y: ",self.y)
            #
    def get_RGB(self):
        return self.RGB

    def get_HSV(self):
        return self.HSV

    def get_click_num(self):
        return self.click_num
# window_name = "Test mouse"
# cv2.namedWindow(window_name)
# img = cv2.imread("photo_2.jpg")
# cv2.imshow(window_name, img)
# event_listener = MouseEvent(window_name, img)
# cv2.waitKey(0)
