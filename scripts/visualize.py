#! /usr/bin/env python

import rospy
from openpose_ros.msg import PersonArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
import numpy as np
from labels import POSE_COCO_L1

bridge = CvBridge()
image = None

def callback(image_msg, people_msg):
    print 'callback'
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)
    for person in people_msg.people:
        for part in person.body_parts:
            cv2.circle(cv_image, (int(part.x), int(part.y)), 3, (0, 0, 255), 1)
        parts = {part.name: (int(part.x), int(part.y)) for part in person.body_parts}
        for p1, p2 in POSE_COCO_L1:
            if p1 in parts and p2 in parts:
                cv2.circle(cv_image, parts[p1], parts[p2], (0, 255, 0), 2)
    
    global image
    image = cv_image

if __name__ == '__main__':
    rospy.init_node('visualize_born')
    image_sub = Subscriber('image', Image)
    people_sub = Subscriber('people', PersonArray)
    sub = ApproximateTimeSynchronizer([image_sub, people_sub], 100, 100)
    sub.registerCallback(callback)
    while not rospy.is_shutdown():
        if image is not None:
            cv2.imshow('OpenPose visualization', image)
        cv2.waitKey(1)
