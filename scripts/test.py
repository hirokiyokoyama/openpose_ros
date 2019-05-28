from dataset import load_coco_dataset, generate_cm, generate_paf
import tensorflow as tf
import numpy as np
import os
import cv2
from nets import pose_net_body_25
from labels import POSE_BODY_25_L2, POSE_BODY_25_L1

if __name__=='__main__':
  FOOT_FILE = '/home/yokoyama/Documents/coco/annotations/person_keypoints_val2017_foot_v1.json'
  BODY_FILE = '/home/yokoyama/Documents/coco/annotations/person_keypoints_val2017.json'
  IMAGE_DIR = '/home/yokoyama/Documents/coco/images/val2017'
  CKPT_FILE = '/home/yokoyama/Documents/pose_iter_584000.ckpt'
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
  
  sigma = 8.
  thickness = 15.
  j = 0
  while True:
    imgfile = os.path.join(IMAGE_DIR, images[j])
    img = cv2.imread(imgfile)
    _inputs = [cv2.resize(img, (368,368))/255.]

    cm, paf = sess.run([end_points['stage2_L2'], end_points['stage4_L1']],
                       {inputs: _inputs})

    cm_gt = generate_cm(img.shape[:2], keypoints[j], sigma=sigma/8, target_shape=(46,46))
    paf_gt = generate_paf(img.shape[:2], affinities[j], thickness=thickness/8, target_shape=(46,46))

    part_inds = [0,2,5]
    limb_ind = 0
    
    _img = cv2.resize(img, cm.shape[2:0:-1])
    cv2.imshow('ConfidenceMap', cv2.resize(np.uint8((_img + cm[0][:,:,part_inds]*255)/2), img.shape[1::-1]))
    cv2.imshow('Background', cv2.resize(np.uint8(cm[0,:,:,-1]*255), img.shape[1::-1]))
    pafimg = np.stack([np.arctan2(-paf[0,:,:,limb_ind*2], paf[0,:,:,limb_ind*2+1])/(2*np.pi)*180+90,
                       np.ones(paf.shape[1:3])*255,
                       np.hypot(paf[0,:,:,limb_ind*2], paf[0,:,:,limb_ind*2+1])*255], 2)
    pafimg = cv2.cvtColor(np.uint8(pafimg), cv2.COLOR_HSV2BGR)
    cv2.imshow('PartAffinityField', cv2.resize(np.uint8(_img*.5 + pafimg*.5), img.shape[1::-1]))

    cv2.imshow('GroundTruthConfidenceMap', cv2.resize(np.uint8((_img + cm_gt[:,:,part_inds]*255)/2), img.shape[1::-1]))
    cv2.imshow('GroundTruthBackground', cv2.resize(np.uint8(cm_gt[:,:,-1]*255), img.shape[1::-1]))
    pafimg = np.stack([np.arctan2(-paf_gt[:,:,limb_ind*2], paf_gt[:,:,limb_ind*2+1])/(2*np.pi)*180+90,
                       np.ones(paf_gt.shape[:2])*255,
                       np.hypot(paf_gt[:,:,limb_ind*2], paf_gt[:,:,limb_ind*2+1])*255], 2)
    pafimg = cv2.cvtColor(np.uint8(pafimg), cv2.COLOR_HSV2BGR)
    cv2.imshow('GroundTruthPartAffinityField', cv2.resize(np.uint8(_img*.5 + pafimg*.5), img.shape[1::-1]))
    
    key = cv2.waitKey()
    if key == ord(' '):
      break
    if key == ord('t'):
      sess.run(train_op, {inputs: _inputs, l2_gt: [cm_gt], l1_gt: [paf_gt]})
    j += 1
