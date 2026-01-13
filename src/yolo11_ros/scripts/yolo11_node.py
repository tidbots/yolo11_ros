#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class Yolo11Node:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO("yolo11n.pt")

        self.sub = rospy.Subscriber(
            "/camera/image_raw",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24
        )

        self.pub = rospy.Publisher(
            "/yolo/image",
            Image,
            queue_size=1
        )

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(img, conf=0.4)

        annotated = results[0].plot()
        out_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

if __name__ == "__main__":
    rospy.init_node("yolo11_node")
    Yolo11Node()
    rospy.spin()
