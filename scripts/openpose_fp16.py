#!/usr/bin/env python

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
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
from functools import partial

class KeyPointDetector:
  def __init__(self, model, ckpt_file, part_names,
               limbs=None, input_size=None):
    self.model = model
    self.ckpt_file = ckpt_file
    if input_size is None:
      self._input_size = None
    else:
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
    if self.input_size is not None:
      shape[1] = self.input_size[0]
      shape[2] = self.input_size[1]
    #self.model.build(tuple(shape))
    self.model(tf.zeros([1,shape[1] or 128,shape[2] or 128,3], dtype=tf.float16))
    self.model.load_weights(self.ckpt_file)
    rospy.loginfo('Network was restored from {}.'.format(self.ckpt_file))

  def finalize(self):
    pass

  @partial(tf.contrib.eager.defun, autograph=False)
  #@tf.function
  def _run(self, image,
           key_point_threshold = tf.constant(0.5)):
    image = tf.cast(image[tf.newaxis], tf.float16) / 255.
    if self.input_size is not None:
      image = tf.image.resize(image, self.input_size)
      image = tf.cast(image, tf.float16)
      
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
    heat_map, affinity, keypoints = map(lambda x: x.numpy(), predictions)
    #scale_x = orig_shape[1]/float(heat_map.shape[2])
    #scale_y = orig_shape[0]/float(heat_map.shape[1])
    scale_x = 1./float(heat_map.shape[2])
    scale_y = 1./float(heat_map.shape[1])
    inlier_lists = []
    for _,y,x,c in keypoints:
      x = x*scale_x + scale_x/2
      y = y*scale_y + scale_y/2
      inlier_lists.append((x,y))

    persons = connect_parts(affinity[0], keypoints[:,1:], self.limbs,
                            line_division=line_division, threshold=affinity_threshold)
    persons = [{self.part_names[k]:inlier_lists[v] \
                for k,v in person.items()} for person in persons]
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

  image_width = rospy.get_param('openpose_image_width', 360)
  image_height = rospy.get_param('openpose_image_height', 270)
  if image_width > 0 and image_height > 0:
    input_size = (image_height, image_width)
  else:
    input_size = None
  
  ckpt_file = os.path.join(pkg, 'data', 'openpose_fp16.ckpt')
  pose_detector = KeyPointDetector(
    pose_net_body_25(tf.float16), ckpt_file, POSE_BODY_25_L2, POSE_BODY_25_L1, input_size=input_size)
  pose_detector.initialize()
  pose_params = {}

  image_sub = rospy.Subscriber('image', Image, callback,
                               queue_size=1, buff_size=1024*1024*4, tcp_nodelay=True)
  people_pub = rospy.Publisher('people', PersonArray, queue_size=1)
  rospy.Service('detect_people', DetectPeople, detect_people)

  srv = Server(KeyPointDetectorConfig, reconf_callback)
  rospy.spin()
