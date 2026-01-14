#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO


class Yolo11Node:
    def __init__(self):
        self.bridge = CvBridge()

        model_path = rospy.get_param("~model", "yolo11n.pt")
        self.model = YOLO(model_path)

        self.sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24
        )

        self.pub = rospy.Publisher(
            "/yolo11/debug_image",
            Image,
            queue_size=1
        )

        rospy.loginfo("YOLO11 node started")

    def image_cb(self, msg):
        # ROS Image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YOLO推論
        results = self.model(frame, verbose=False)[0]

        # YOLOの描画ユーティリティ
        debug_img = results.plot()  # BGR ndarray

        # OpenCV -> ROS Image
        out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        out_msg.header = msg.header

        self.pub.publish(out_msg)


def main():
    rospy.init_node("yolo11_node")
    Yolo11Node()
    rospy.spin()


if __name__ == "__main__":
    main()

