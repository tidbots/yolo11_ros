#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from vision_msgs.msg import Detection2DArray


class SemanticFilterNode:
    """
    Semantic filter for YOLO Detection2DArray (ROS1 Noetic)

    - Input : /yolo11/detections (Detection2DArray)
    - Output: /yolo11/filtered_detections (Detection2DArray)

    Filtering is done by:
      - allowed class IDs (YOLO class index)
      - minimum confidence score
    """

    def __init__(self):
        # parameters
        # example: allow_ids = [0]  # person
        # example: allow_ids = [39] # bottle (YOLO COCO)
        self.allow_ids = rospy.get_param("~allow_ids", [])
        self.min_score = rospy.get_param("~min_score", 0.0)

        self.sub = rospy.Subscriber(
            "/yolo11/detections",
            Detection2DArray,
            self.cb,
            queue_size=1
        )

        self.pub = rospy.Publisher(
            "/yolo11/filtered_detections",
            Detection2DArray,
            queue_size=1
        )

        rospy.loginfo(
            "SemanticFilterNode started | allow_ids=%s min_score=%.2f",
            self.allow_ids,
            self.min_score
        )

    def cb(self, msg: Detection2DArray):
        out = Detection2DArray()
        out.header = msg.header

        for det in msg.detections:
            if not det.results:
                continue

            hypo = det.results[0]  # YOLO は 1 detection = 1 hypothesis
            cls_id = hypo.id
            score = hypo.score

            # ---- class ID filter ----
            if self.allow_ids and cls_id not in self.allow_ids:
                continue

            # ---- confidence filter ----
            if score < self.min_score:
                continue

            out.detections.append(det)

        self.pub.publish(out)


def main():
    rospy.init_node("semantic_filter_node")
    SemanticFilterNode()
    rospy.spin()


if __name__ == "__main__":
    main()

