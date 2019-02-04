#! /usr/bin/env python

import rospy
from openpose_ros.msg import PersonArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
import numpy as np
#import labels

bridge = CvBridge()

def callback(image_msg, people_msg):
    print 'callback'
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)
    for person in people_msg.people:
        print person
        r = int(np.random.randint(255))
        g = int(np.random.randint(255))
        b = int(np.random.randint(255))
        for part in person.body_parts:
            cv2.circle(cv_image, (int(part.x), int(part.y)), 5, (r, g, b), 2)

    image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
    image_pub.publish(image_message)

    #cv2.imshow('cv_image', cv_image)
    #cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('visualize_born')
    image_sub = Subscriber('/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image)
    people_sub = Subscriber('/people', PersonArray)
    sub = ApproximateTimeSynchronizer([image_sub, people_sub], 100, 100)
    image_pub = rospy.Publisher('/bone_image', Image, queue_size=1)
    sub.registerCallback(callback)
    rospy.spin()
