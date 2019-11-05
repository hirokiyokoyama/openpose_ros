#!/usr/bin/env python

import cv2
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import rospy
import threading
import rospkg
import os
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server

from openpose_ros.msg import Person, PersonArray, KeyPoint
from openpose_ros.srv import DetectPeople, DetectPeopleResponse
from openpose_ros.msg import SparseTensor
from openpose_ros.cfg import KeyPointDetectorConfig
from std_srvs.srv import Empty, EmptyResponse
from nets_v2 import non_maximum_suppression, connect_parts, pose_net_body_25
from labels import POSE_BODY_25_L1, POSE_BODY_25_L2

class KeyPointDetector:
  def __init__(self, model, ckpt_file, part_names,
               limbs=None, input_size=None):
    self.model = model
    self.ckpt_file = ckpt_file
    self._input_size = tuple(input_size)
    self.part_names = part_names
    if limbs is not None:
      self.limbs = [(part_names.index(p), part_names.index(q)) for p, q in limbs]
    else:
      self.limbs = None
      
  @property
  def input_size(self):
    return self._input_size

  def initialize(self):
    shape = [None,None,None,3]
    if self.input_shape is not None:
      shape[1] = self.input_size[0]
      shape[2] = self.input_size[1]
    self.model.build(tuple(shape))
    model.load_weights(self.ckpt_file)
    rospy.loginfo('Network was restored from {}.'.format(self.ckpt_file))

  def finalize(self):
    pass

  @tf.function
  def _run(self, image,
           key_point_threshold = tf.constant(0.5)):
    image = tf.cast(image[tf.newaxis], tf.float16) / 255.
    if self.input_size:
      image = tf.image.resize(image, self.input_size)
      
    heat_map, affinity = self.model(image)
    keypoints = non_maximum_suppression(
      heat_map[:,:,:,:-1], threshold=key_point_threshold)
    return heat_map, affinity, keypoints

  def detect_keypoints(self, image,
                       key_point_threshold=0.5,
                       affinity_threshold=0.2,
                       line_division=15):
    orig_shape = image.shape
    rospy.loginfo('Start processing.')
    predictions = self._run(tf.constant(image), tf.constant(key_point_threshold))
    rospy.loginfo('Done.')
    heat_map, affinity, keypoints = predictions
    scale_x = orig_shape[1]/float(heat_map.shape[2])
    scale_y = orig_shape[0]/float(heat_map.shape[1])
    inlier_lists = []
    for _,y,x,c in keypoints:
      x = x*scale_x + scale_x/2
      y = y*scale_y + scale_y/2
      inlier_lists.append((x,y))

    if affinity is not None:
      persons = connect_parts(affinity[0], keypoints[:,1:], self.limbs,
                              line_division=line_division, threshold=affinity_threshold)
      persons = [{self.part_names[k]:inlier_lists[v] \
                  for k,v in person.items()} for person in persons]
    else:
      persons = [{self.part_names[c]:inliers \
                  for (_,_,_,c), inliers in zip(keypoints, inlier_lists)}]
    return persons

def callback(data):
  if not people_pub.get_num_connections():
    return
  try:
    cv_image = bridge.imgmsg_to_cv2(data, 'rgb8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return
  persons = pose_detector.detect_keypoints(cv_image, **pose_params)
  msg = PersonArray(header=data.header)
  msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                   for k,(x,y) in p.items()]) \
                for p in persons]
  people_pub.publish(msg)

def detect_people(req):
  try:
    cv_image = bridge.imgmsg_to_cv2(req.image, 'rgb8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return None
  persons = pose_detector.detect_keypoints(cv_image, **pose_params)
  msg = DetectPeopleResponse()
  msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                   for k,(x,y) in p.items()]) \
                for p in persons]
  return msg

def reconf_callback(config, level):
  for key in ['key_point_threshold', 'affinity_threshold', 'line_division']:
    pose_params[key] = config[key]
  return config

if __name__ == '__main__':
  pkg = rospkg.rospack.RosPack().get_path('openpose_ros')
  bridge = CvBridge()
  rospy.init_node('openpose')

  ckpt_file = os.path.join(pkg, '_data', 'openpose_fp16.ckpt')
  pose_detector = KeyPointDetector(
    pose_net_body_25(tf.float16), ckpt_file, input_size=(480,640))
  pose_detector.initialize()
  pose_params = {}

  image_sub = rospy.Subscriber('image', Image, callback)
  people_pub = rospy.Publisher('people', PersonArray, queue_size=1)
  rospy.Service('detect_people', DetectPeople, detect_people)

  srv = Server(KeyPointDetectorConfig, reconf_callback)
  rospy.spin()
