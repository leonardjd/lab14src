import rospy
import cv2
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge


class Camera:
    def __init__(self, img_topic="/camera/image_rect_color", compressed=False):
        self.image_received = False
        self.compressed = compressed

        self.image_sub = rospy.Subscriber(
            img_topic, Image, self.callback, queue_size=1)
        print("Finish initialize camera1")

    def callback(self, data):
        if self.compressed == False:
            bridge = CvBridge()
            self.img = bridge.imgmsg_to_cv2(
                data, desired_encoding='passthrough')
            self.image_received = True
        else:
            np_arr = np.fromstring(data.data, np.uint8)
            self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.image_received = True

    def check_received(self):
        return self.image_received

    def get_frame(self):
        if self.image_received:
            return self.img

    def save_frame(self, img_title):
        if self.image_received:
            print(cv2.imwrite(img_title, self.img))
