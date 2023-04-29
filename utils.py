import cv2
import numpy as np
from MouseEvent import MouseEvent
import sys
import math
from sklearn.linear_model import LinearRegression, RANSACRegressor


def img_preprocessing(img_bgr, rotate = True):
    """
    This function rotate 90 degree clockwise and change BGR to RGB colorspace
    """
    if rotate:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_bgr, img_rgb

def detect_black_line(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (57,64,0),(78,132,86))
    return mask

def apply_homography(frame, H):
    img_size = frame.shape[:2]
    return cv2.warpPerspective(frame, H, (img_size[1], img_size[0]), cv2.INTER_LINEAR)

def histogram_equalize(img):
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))  
    return equ

def find_intersection(coefficients1, coefficients2):
    '''
    This function will return the x-y value of the intersection of the given lines
    :param coefficients_list: Shape (2,2). The a, b coefficients (in order) of 2 line equation
    :return: the x-y value of the intersection of the given lines
    '''

    a1, b1 = coefficients1
    a2, b2 = coefficients2

    x = (b2 - b1) / ((a1 - a2) + 0.001)
    y = a1 * x + b1

    return int(x), int(y)

def get_sample_points(img_rgb, data_point_num):
    '''
    Pop up the window to select all the point on the desired line to collect it RGB value
    :param img_rgb: cv2 Object (RGB format)
    :param data_point_num: number of data points you want to get
    :return: the rgb value of all clicked points
    '''

    window_name = "Test mouse"
    cv2.namedWindow(window_name)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    event_listener = MouseEvent(window_name, img_rgb)

    rgb_arr = np.zeros((data_point_num, 3))
    click_num = 1
    while True:
        cv2.imshow(window_name, img_bgr)
        if click_num == event_listener.get_click_num():
            rgb_arr[click_num - 1] = event_listener.get_RGB()

            print("Got {} data point(s)".format(click_num))
            click_num += 1
        k = cv2.waitKey(1)
        if k == ord('q'):
            sys.exit()
        if click_num > data_point_num:
            cv2.destroyAllWindows()
            break

    return rgb_arr

def get_stats_from_data_points(rgb_arr):
    '''
    Get the mean and standard deviation from the collected rgb value
    :param rgb_arr: the rgb value of all clicked points, result from get_sample_points()
    :return: the mean and standard deviation of the collected rgb value
    '''
    intensity = np.sum(rgb_arr, 1, keepdims=True)

    # Normalize
    rgb_arr = np.divide(rgb_arr, intensity)

    mean = np.mean(rgb_arr, 0)
    std = np.std(rgb_arr, 0)
    print("Mean: {}".format(mean))
    print("Standard deviation: {}".format(std))

    return mean, std

def normalize_img(img_rgb):
    '''
    Normalize RGB image
    :param img_rgb: input image
    :return: normalize image
    '''

    sum_temp = np.sum(img_rgb, 2, keepdims=True) + 0.0001  # Avoid dvide by zero
    normal_img = np.divide(img_rgb, sum_temp)

    return normal_img

def find_lane(normal_img, rgb_mean, rgb_std, line_type, centerline_coef):
    '''
    Return a numpy array of mask of detecting lanes. This will run a Gaussian probability
     (with the input mean and standard deviation) on the input images and find the pixels
     having the closest color with the input mean and standard deviation
    :param normal_img: the normalize image that you want to detect lanes inside it
    :param rgb_mean: the mean value of the RGB range you want to detect
    :param rgb_std: the standard deviation value of the RGB range you want to detect
    :return: a numpy array of mask of detecting lanes. Shape: (image height, image width)
    '''
    img_shape = np.shape(normal_img)
    
    #Threshold the prob_arr to binary
    if line_type == 1:
        threshold =  1.9 * rgb_std
    elif line_type == 2:
        threshold = 2.1 * rgb_std
    elif line_type == 3:
        threshold = rgb_std
    
    prob_arr = (abs(normal_img - rgb_mean) <= threshold) * 255
    prob_arr = np.prod(prob_arr, axis = 2, keepdims=True)
    prob_arr = prob_arr.astype(np.uint8)
    
    #Morphological transformation to close small holes
    prob_arr = cv2.medianBlur(prob_arr, 7)

    kernel = np.ones((7, 7))
    prob_arr = cv2.morphologyEx(prob_arr, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(prob_arr,kernel,iterations = 1)
    if line_type == 1:
        coef = linear_regression(prob_arr, "RANSAC")
    if line_type == 2:
        coef = linear_regression(prob_arr, "RANSAC")
        if coef[1] < centerline_coef[1]:
            prob_arr = eliminate_pixels_along_line(prob_arr, coef, margin = 75)
            coef = linear_regression(prob_arr, "RANSAC")

    if line_type == 3:
        return prob_arr
    cv2.imshow(f"Mask {line_type}", prob_arr.astype(np.uint8))

    return coef

def linear_regression(lane_mask, regressor_type = "linear"):
    '''
    Use the linear regression to find the line of best fit for the input line
    In other word, symbolize the whole lane area with a line equation
    :param lane_mask: a numpy array of mask of detected lanes.
    :return the slope and y-intercept of the line representation of the lane
    '''
    line_pixel_coor = np.where(lane_mask == 255)
    line_pixel_coor = np.expand_dims(line_pixel_coor, 2)
    if np.shape(line_pixel_coor)[1] == 0:
        #Edge case, will results error, wait for the enxt frame
        return (0, 0)
    if regressor_type == "linear":
        regressor = LinearRegression()
        regressor.fit(line_pixel_coor[1], line_pixel_coor[0])
        return (regressor.coef_[0], regressor.intercept_[0])
    else:
        regressor = RANSACRegressor()
        regressor.fit(line_pixel_coor[1], line_pixel_coor[0])
        return (regressor.estimator_.coef_[0], regressor.estimator_.intercept_[0])
def eliminate_pixels_along_line(img, line_coef, margin = 5):
    '''
    This function will turn all pixel near the line characterized by the argument line_coef (with the margin in the argument)
    :param: img a 2D numpy array
    :param: line_coef a tuple (slope, y_intercept)
    :param: margin the width of the area will be eliminated
    '''
    a, b = line_coef
    x1 = -1000
    y1 = int(a*x1+b)

    x2= 1000
    y2 = int(a*x2+b)
    img = cv2.line(img, (x1, y1), (x2, y2), 0, margin)
    return img


def find_centerline(coefficients1, coefficients2, img_shape):
    """
    Find the slope and the y-intercept of the centerline of 2 lines in the argument
    :param coefficients1 slope and y-intercept of the first line
    :param coefficients2 slope and y-intercept of the second line
    :return slope and y-intercept of the center line
    """
    x1 = img_shape[1]
    x2 = x1+50
    a1, b1 = coefficients1
    a2, b2 = coefficients2

    y_mid1 = (get_y(x1, a1, b1) + get_y(x1, a2, b2)) / 2.0
    y_mid2 = (get_y(x2, a1, b1) + get_y(x2, a2, b2)) / 2.0

    slope = (y_mid1 - y_mid2) / (x1 - x2)
    y_intercept = y_mid1 - slope * x1
    return slope, y_intercept

def get_y(x, a, b):
    return a*x + b

def find_d1_d2(coefficients1, coefficients2, x_intersect, y_intersect, img_shape):
    """
    Find d1, d2. Please refer to illus.pdf to understand what d1 and d2
    """
    y0 = img_shape[1]
    a1 = coefficients1[0]
    a2 = coefficients2[0]

    d1 = y0*1.0/a1
    d2 = -y0*1.0/a2

    return d1, d2
def get_optimal_speed(line_coefficients_list, intersection, img_shape_rotate, i):
    #This function will determine the normalized optimal linear and angular speed based on the current location of the led

    COEFFICIENT1 = 0.004
    COEFFICIENT2 = 0.35

    x_intersect, y_intersect = intersection
    center_slope, _ = find_centerline(line_coefficients_list[0], line_coefficients_list[1], img_shape_rotate)

    angular_vel = -(y_intersect - img_shape_rotate[0]/2) * COEFFICIENT1 + center_slope  * COEFFICIENT2
    term1 = -(y_intersect - img_shape_rotate[0]/2)
    term1_ = term1*COEFFICIENT1
    term2 = center_slope
    term2_ = term2*COEFFICIENT2
    print("Term1: {}, Term2: {}".format(term1, term2))
    print("Term1_after: {}, Term2_after: {}".format(term1_, term2_))
    if angular_vel > 0.3:
        angular_vel = 0.3
    if angular_vel < -0.3:
        angular_vel = -0.3

    linear_vel = 0.07
    if True:
        print("Linear_vel " + str(linear_vel))
        print("Angular_vel " + str(angular_vel))

    return linear_vel, angular_vel
