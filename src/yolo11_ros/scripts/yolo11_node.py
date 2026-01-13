#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D

from cv_bridge import CvBridge
from ultralytics import YOLO

import image_transport
import cv2

class Yolo11RosNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.it = image_transport.ImageTransport(rospy)

        # params
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.model_path  = rospy.get_param("~model", "yolo11n.pt")
        self.conf_thres  = float(rospy.get_param("~conf", 0.35))
        self.iou_thres   = float(rospy.get_param("~iou", 0.45))
        self.max_det     = int(rospy.get_param("~max_det", 50))
        self.pub_annot   = rospy.get_param("~publish_annotated", True)

        # model
        rospy.loginfo(f"[yolo11] loading model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # pubs/subs
        self.sub = rospy.Subscriber(self.image_topic, Image, self.cb, queue_size=1, buff_size=2**24)
        self.pub_det = rospy.Publisher("detections", Detection2DArray, queue_size=10)

        if self.pub_annot:
            self.pub_img = self.it.advertise("image", 1)

        rospy.loginfo(f"[yolo11] subscribe: {self.image_topic}")
        rospy.loginfo("[yolo11] publish:  ~detections (vision_msgs/Detection2DArray), ~image (annotated)")

    def cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"[yolo11] cv_bridge error: {e}")
            return

        # Ultralytics expects RGB or BGR OK; we'll keep BGR as numpy
        try:
            results = self.model.predict(
                source=img,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                verbose=False
            )
        except Exception as e:
            rospy.logerr(f"[yolo11] inference error: {e}")
            return

        r0 = results[0]
        det_msg = Detection2DArray()
        det_msg.header = msg.header

        # boxes: xyxy, cls, conf
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            conf = r0.boxes.conf.cpu().numpy()
            cls  = r0.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                d = Detection2D()
                d.header = msg.header

                # bbox center + size
                bb = BoundingBox2D()
                bb.center = Pose2D(x=float((x1+x2)/2.0), y=float((y1+y2)/2.0), theta=0.0)
                bb.size_x = float(x2 - x1)
                bb.size_y = float(y2 - y1)
                d.bbox = bb

                h = ObjectHypothesisWithPose()
                h.id = int(c)              # class id
                h.score = float(s)         # confidence
                d.results.append(h)

                det_msg.detections.append(d)

        self.pub_det.publish(det_msg)

        if self.pub_annot:
            try:
                annotated = r0.plot()  # BGR uint8
                out = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
                out.header = msg.header
                self.pub_img.publish(out)
            except Exception as e:
                rospy.logwarn(f"[yolo11] annotate publish error: {e}")

def main():
    rospy.init_node("yolo11_node")
    Yolo11RosNode()
    rospy.spin()

if __name__ == "__main__":
    main()
