#! /usr/bin/python3

from calendar import c
from code import interact
import numpy as np
import cv2
from Camera import Camera
import os
import rospy
from utils import *
import os.path
from std_msgs.msg import String
import sys

if __name__ == "__main__":
    mode = sys.argv[1]
    print(f"Entering mode: {mode}")
    stop_pub = rospy.Publisher("stop_pub", String, queue_size=0)
    rospy.init_node("homography")
    print("here")
    camera = Camera()
    #print(mode)
    print("here2")

    while camera.check_received() == False:
        pass
    img_bgr = camera.get_frame()
    IMG_SIZE = img_bgr.shape[:2]
    corners = None
    print(f"Enter mode: {mode}")
    if mode == "calib":
        print("here4")
        temp_choice = input(
            "Do you want to use the homography/extrinsic calibration value if it already exist? [y/n]")
        if not os.path.exists('homography_params.npy') or temp_choice == 'n':
            print("Please put the checkerboard on the same plane as the road and in the lower half of the camera sight.")
            print("Make sure it is reasonably in the center of the view")
            print(
                "Please type 's' when you think all the corners of the checkerboard is detected.")
            while True:
                img_bgr = camera.get_frame()

                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                checkerboard = img_bgr[IMG_SIZE[0]//5 * 3:,
                                       IMG_SIZE[1] // 7: IMG_SIZE[1] // 7 * 6]
                y_offset = IMG_SIZE[0] - checkerboard.shape[0]
                x_offset = (IMG_SIZE[1] - checkerboard.shape[1]) // 2
                checkerboard_gray = cv2.cvtColor(
                    checkerboard, cv2.COLOR_BGR2GRAY)
                retval, corners = cv2.findChessboardCorners(
                    checkerboard_gray, (6, 8), None)

                img_copy = np.copy(img_bgr)

                if retval:
                    criteria = (cv2.TERM_CRITERIA_EPS +
                                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(
                        checkerboard_gray, corners, (7, 7), (-1, -1), criteria)

                    corners = np.squeeze(corners).astype(np.uint16)
                    for i in range(len(corners)):
                        x = corners[i][0] = corners[i][0] + x_offset
                        y = corners[i][1] = corners[i][1] + y_offset
                        img_copy = cv2.circle(
                            img_copy, (x, y), 3, (0, 0, 255), -1)

                cv2.imshow("Checker board detection", img_copy)
                print("waiting for key")
                k = cv2.waitKey(1)
                if k == ord('s'):
                    cv2.destroyAllWindows()
                    break
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit()
            corners_proj = np.array(np.meshgrid(
                np.arange(260, 420, 20), np.arange(400, 280, -20)))
            corners_proj = np.reshape(corners_proj, (2, -1), order='F').T

            H, _ = cv2.findHomography(corners, corners_proj, cv2.RANSAC, 5.0)
            np.save('homography_params.npy', H)

    if mode == "get_black_rgb":
        if not os.path.exists('homography_params.npy'):
            rospy.signal_shutdown("You haven't done extrinsic calibration")

        H = np.load('homography_params.npy', allow_pickle=True)
        print(f"The homography matrix is{H}")

        temp_choice = input(
            "Do you want to use the old black rgb value if it already exist? [y/n]")
        if not os.path.exists('rgb_stat_black.npy') or temp_choice == 'n':
            img_bgr = camera.get_frame()
            img_bgr = histogram_equalize(img_bgr)
            img_bgr = apply_homography(img_bgr, H)

            # This variable line_rgb_stats will save the rgb value statistics of two outer lines
            # Format is [(rgb_mean_1st_line, rgb_std_1st_line),(rgb_mean_2nd_line, rgb_std_2nd_line)]
            sample_num = 30
            rgb_arr = get_sample_points(img_bgr, sample_num)
            print("Please click {} points from black line".format(sample_num))
            black_rgb_mean, black_rgb_std = get_stats_from_data_points(rgb_arr)

            # Save the array into csv file
            np.save('rgb_stat_black.npy', [black_rgb_mean, black_rgb_std])

    if mode == "action":
        print("Made it to action")
        if not os.path.exists('homography_params.npy'):
            rospy.signal_shutdown("You haven't done extrinsic calibration")
        H = np.load('homography_params.npy', allow_pickle=True)
        print(f"The homography matrix is{H}")

        if not os.path.exists('rgb_stat_black.npy'):
            rospy.signal_shutdown(
                "You haven't collected RGB value of the black line")
        black_rgb_mean, black_rgb_std = np.load('rgb_stat_black.npy')
        while True:
            img_bgr = camera.get_frame()
            img_bgr = histogram_equalize(img_bgr)
            img_bgr = cv2.warpPerspective(
                img_bgr, H, (IMG_SIZE[1], IMG_SIZE[0]))

            prob_arr = find_lane(img_bgr, black_rgb_mean,
                                 black_rgb_std, 3, None)
            prob_arr = cv2.medianBlur(prob_arr, 11)

            cnts, hierachy = cv2.findContours(
                prob_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) != 0:
                cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(img_bgr, [cnt], 0, (0, 0, 255), 4)

                M = cv2.moments(cnt)
                cY = int(M["m01"] / (M["m00"] + 0.0001))
                img_bgr = cv2.putText(img_bgr, str(cY), (200, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                      3, (0, 0, 0), 3, cv2.LINE_AA)
                if cY > 330:
                    img_bgr = cv2.putText(img_bgr, 'Stop!!!', (200, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                          3, (0, 0, 0), 3, cv2.LINE_AA)
                    stop_pub.publish("stop")

            cv2.imshow("black line", img_bgr)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        cv2.destroyAllWindows()
