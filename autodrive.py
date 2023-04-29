#! /usr/bin/python3

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from utils import *
from motor_control import Motor
from Camera import Camera
import os
from time import sleep


class stopSubscriber:
    def __init__(self):
        self.stop = False
        rospy.Subscriber("/stop_pub", String, self.stop_callback)

    def stop_callback(self, msg):
        self.stop = True

    def getCmd(self):
        return self.stop


if __name__ == "__main__":
    rospy.init_node("AutoDrive")
    camera = Camera()
    motor = Motor()

    while camera.check_received() == True:
        pass
    print("Finished checking connection to camera")

    temp_choice = input(
        "Do you want to use the old rgb value if it already exist? [y/n]")
    if not os.path.exists('rgb_stat.npy') or temp_choice == 'n':
        img_bgr = camera.get_frame()
        img_bgr, img_rgb = img_preprocessing(img_bgr, False)

        # This variable line_rgb_stats will save the rgb value statistics of two outer lines
        # Format is [(rgb_mean_1st_line, rgb_std_1st_line),(rgb_mean_2nd_line, rgb_std_2nd_line)]
        line_rgb_stats = []

        for i in range(2):
            sample_num = 30
            rgb_arr = get_sample_points(img_rgb, sample_num)
            print("Please click {} points from line {}".format(sample_num, i + 1))
            rgb_mean, rgb_std = get_stats_from_data_points(rgb_arr)
            line_rgb_stats.append((rgb_mean, rgb_std))

        # Save the array into csv file
        np.save('rgb_stat.npy', line_rgb_stats)
    else:
        line_rgb_stats = np.load('rgb_stat.npy')[:2]
    k = 0
    stopSub = stopSubscriber()
    while True:
        if stopSub.getCmd() == True:
            motor.go_forward()
            rospy.sleep(4)
            motor.stop()
            cv2.destroyAllWindows()
            break

        k = cv2.waitKey(1)
        if k == ord('q'):
            motor.stop()
            cv2.destroyAllWindows()
            break
        img_bgr = camera.get_frame()
        img_bgr, img_rgb = img_preprocessing(img_bgr)
        img_shape_rotate = np.shape(img_bgr)

        # Find 2 lines and add the found coeffiecients to the coefficient_list
        coefficients_list = []
        for i in range(2):  # Because we have 2 lines
            rgb_mean, rgb_std = line_rgb_stats[i]
            normal_img = normalize_img(img_rgb)
            centerline_coef = None
            if i == 1:
                centerline_coef = coefficients_list[0]
            coefficients = find_lane(
                normal_img, rgb_mean, rgb_std, i+1, centerline_coef)

            coefficients_list.append(coefficients)

        print(coefficients_list)
        x_intersect, y_intersect = find_intersection(
            coefficients_list[0], coefficients_list[1])
        print("x_intersect and y_intersect: ", end='')
        print(x_intersect, y_intersect)

        # Get the optimal speed based on the intersection of two outer lines
        linear_vel, angular_vel = get_optimal_speed(
            coefficients_list, (x_intersect, y_intersect), img_shape_rotate, k)
        print("---------------------------------------")
        # motor.move_with_coor(li near_vel, angular_vel)
        motor.move_with_coor(linear_vel, angular_vel)
        k += 1

        for i in range(2):
            slope, y_intercept = coefficients_list[i]
            x1 = -1000
            y1 = int(x1 * slope + y_intercept)

            x2 = 1000
            y2 = int(x2 * slope + y_intercept)
            temp_img = cv2.line(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        temp_img = cv2.circle(
            img_bgr, (x_intersect, y_intersect), 7, (255, 255, 0), -1)
        temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("intersection", temp_img)
