#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import Detection2DArray, Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance

from ultralytics import YOLO


class Yolo11Node:
    def __init__(self):
        self.bridge = CvBridge()

        # parameters
        self.model_path = rospy.get_param("~model", "yolo11n.pt")
        self.conf_th = rospy.get_param("~conf", 0.4)

        # YOLO model
        self.model = YOLO(self.model_path)

        # subscriber
        self.sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24
        )

        # publishers
        self.pub_debug = rospy.Publisher(
            "/yolo11/debug_image",
            Image,
            queue_size=1
        )

        self.pub_det = rospy.Publisher(
            "/yolo11/detections",
            Detection2DArray,
            queue_size=1
        )

        rospy.loginfo("YOLO11 node started (ROS1 Noetic FINAL-CORRECT)")

    def image_cb(self, msg):
        # --------------------------------------
        # Image conversion
        # --------------------------------------
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # --------------------------------------
        # YOLO inference
        # --------------------------------------
        results = self.model(
            frame,
            conf=self.conf_th,
            verbose=False
        )[0]

        # ======================================
        # (A) Debug image (そのまま)
        # ======================================
        debug_img = results.plot()
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        debug_msg.header = msg.header
        self.pub_debug.publish(debug_msg)

        # ======================================
        # (B) Detection2DArray (Noetic 正解)
        # ======================================
        det_array = Detection2DArray()
        det_array.header = msg.header

        if results.boxes is not None:
            for box in results.boxes:
                det = Detection2D()

                # bbox
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det.bbox.center.x = (x1 + x2) / 2.0
                det.bbox.center.y = (y1 + y2) / 2.0
                det.bbox.size_x = (x2 - x1)
                det.bbox.size_y = (y2 - y1)

                # hypothesis (ROS1 Noetic)
                cls_id = int(box.cls[0])
                score = float(box.conf[0])

                hypo = ObjectHypothesisWithPose()
                hypo.id = cls_id
                hypo.score = score

                # pose は必須 → identity を入れる
                pose = PoseWithCovariance()
                pose.pose.orientation.w = 1.0
                hypo.pose = pose

                det.results.append(hypo)
                det_array.detections.append(det)

        self.pub_det.publish(det_array)


def main():
    rospy.init_node("yolo11_node")
    Yolo11Node()
    rospy.spin()


if __name__ == "__main__":
    main()

