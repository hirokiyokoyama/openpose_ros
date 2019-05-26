from dataset import load_coco_dataset, generate_cm, generate_paf
import tensorflow as tf
import numpy as np
import os
import cv2
from nets import pose_net_body_25
from labels import POSE_BODY_25_L2, POSE_BODY_25_L1

def generate_batch(image_dir, images, keypoints, affinities, shape, sigma=8., thickness=15.):
  n = len(images)
  map_shape = (shape[0]/8, shape[1]/8)
  inputs = np.zeros((n,)+shape+(3,))
  cm = np.zeros((n,)+map_shape+(26,))
  paf = np.zeros((n,)+map_shape+(52,))
  
  for i, (img, kp, af) in enumerate(zip(images, keypoints, affinities)):
    imgfile = os.path.join(image_dir, img)
    img = cv2.imread(imgfile)
    inputs[i] = cv2.resize(img, shape) / 255.
    cm[i] = generate_cm(img.shape[:2], kp, sigma=sigma, target_shape=map_shape)
    paf[i] = generate_paf(img.shape[:2], af, thickness=thickness, target_shape=map_shape)
  return inputs, cm, paf

if __name__=='__main__':
  FOOT_FILE = '/media/psf/Home/Documents/coco/annotations/person_keypoints_val2017_foot_v1.json'
  BODY_FILE = '/media/psf/Home/Documents/coco/annotations/person_keypoints_val2017.json'
  IMAGE_DIR = '/media/psf/Home/Documents/coco/images/val2017'
  CKPT_FILE = '/home/yokoyama/Documents/openpose_data/pose_iter_584000.ckpt'
  images, keypoints, affinities = load_coco_dataset(
    BODY_FILE, FOOT_FILE,
    keypoint_names=POSE_BODY_25_L2,
    limb_names=POSE_BODY_25_L1)

  with tf.Graph().as_default() as graph:
    inputs = tf.placeholder(tf.float32, [None, 368, 368, 3])
    end_points = pose_net_body_25(inputs)
    saver = tf.train.Saver()

    batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
    l2_gt = tf.placeholder(tf.float32, [None, 46, 46, 26])
    l1_gt = tf.placeholder(tf.float32, [None, 46, 46, 52])
    
    l2_losses = []
    for stage in range(2):
      cm = end_points['stage{}_L2'.format(stage+1)]
      l2_losses.append(tf.nn.l2_loss(tf.square(cm - l2_gt)))
    l2_loss = tf.reduce_mean(l2_losses) / batch_size

    l1_losses = []
    for stage in range(4):
      paf = end_points['stage{}_L1'.format(stage+1)]
      l1_losses.append(tf.nn.l2_loss(tf.square(paf - l1_gt)))
    l1_loss = tf.reduce_mean(l1_losses) / batch_size

    loss = (l1_loss + l2_loss)/2
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
      1e-4, global_step, float(len(images))/batch_size, 0.5, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
    train_op = opt.minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
  sess = tf.Session(graph=graph)
  sess.run(init_op)
  saver.restore(sess, CKPT_FILE)
  
  sigma = 8./8.
  thickness = 15./8.
  j = 0
  while True:
    x, cm_gt, paf_gt = generate_batch(IMAGE_DIR,
                                      images[j:j+4], keypoints[j:j+4], affinities[j:j+4],
                                      (368,368),
                                      sigma=sigma, thickness=thickness)
    sess.run(train_op, {inputs: x, l2_gt: cm_gt, l1_gt: paf_gt})
    j += 4
