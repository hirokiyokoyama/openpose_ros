#! /usr/bin/env python

import rospy
from openpose_ros.srv import DetectPeople
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from labels import POSE_BODY_25_L1

bridge = CvBridge()
image = None

limb_colors = [
  (255,0,255),
  (255,255,0),
  (255,128,0),
  (128,255,0),
  (0,255,128),
  (0,255,255),
  (0,128,255),
  (128,255,0),
  (255,255,0),
  (255,128,0),
  (255,255,128),
  (0,255,128),
  (0,255,255),
  (0,128,255),
  (128,255,255),
  (255,128,255),
  (255,255,128),
  (128,255,255),
  (255,128,128),
  (128,128,255),
  (0,0,255),
  (0,0,255),
  (0,0,255),
  (255,0,0),
  (255,0,0),
  (255,0,0),
]

def draw_peoplemsg(image, people):
  for person in people.people:
    points = {}
    for part in person.body_parts:
      k = part.name
      v = (int(part.x*image.shape[1]), int(part.y*image.shape[0]))
      points[k] = v
      cv2.circle(image, v, 3, (0, 255, 0), 1)
    for (p1, p2), color in zip(POSE_BODY_25_L1, limb_colors):
      if p1 in points and p2 in points:
        cv2.line(image, points[p1], points[p2], color, 3)

def callback(image_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    people_msg = detect_people(cv_image)
    cv_image = cv_image.copy()
    draw_peoplemsg(cv_image, people_msg)
    
    global image
    image = cv_image

if __name__ == '__main__':
    rospy.init_node('visualize_openpose')
    detect_people = rospy.ServiceProxy('detect_people', DetectPeople)
    image_sub = rospy.Subscriber('image', Image, callback)
    while not rospy.is_shutdown():
        if image is not None:
            cv2.imshow('OpenPose visualization', image)
        cv2.waitKey(1)
