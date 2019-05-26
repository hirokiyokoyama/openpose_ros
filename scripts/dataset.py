import cv2
import json
import os
import numpy as np

def generate_cm(img_shape, keypoints, sigma=8., target_shape=None):
  if target_shape is None:
    target_shape = img_shape
  rx = target_shape[1] / float(img_shape[1])
  ry = target_shape[0] / float(img_shape[0])
  
  cm = np.zeros(target_shape+(keypoints.shape[1]+1,))

  # prepare gaussian patch
  patch_size = int(np.ceil(3*sigma))
  space = np.linspace(-3*sigma, 3*sigma, patch_size * 2 + 1)
  X, Y = np.meshgrid(space, space)
  gaussian_patch = np.exp(-(X**2+Y**2)/(2*sigma**2))
  
  for kp in keypoints:
    for i, (x, y) in enumerate(kp):
      if x+y == 0:
        continue
      x *= rx
      y *= ry
      _x0 = int(min(max(x-patch_size, 0), cm.shape[1]-1))
      _x1 = int(min(max(x+patch_size, 0), cm.shape[1]-1))
      __x0 = int(max(_x0 - (x-patch_size), 0))
      __x1 = __x0 + _x1 - _x0
      _y0 = int(min(max(y-patch_size, 0), cm.shape[0]-1))
      _y1 = int(min(max(y+patch_size, 0), cm.shape[0]-1))
      __y0 = int(max(_y0 - (y-patch_size), 0))
      __y1 = __y0 + _y1 - _y0
      cm[_y0:_y1+1,_x0:_x1+1,i] = np.maximum(
        cm[_y0:_y1+1,_x0:_x1+1,i], gaussian_patch[__y0:__y1+1,__x0:__x1+1])
  cm[:,:,-1] = 1. - cm[:,:,:-1].max(axis=2)
  return cm

def generate_paf(img_shape, affinities, thickness=12., target_shape=None):
  if target_shape is None:
    target_shape = img_shape
  rx = target_shape[1] / float(img_shape[1])
  ry = target_shape[0] / float(img_shape[0])
  
  paf = np.zeros(target_shape+(affinities.shape[1]*2,))
  npaf = np.zeros(paf.shape, np.int32)
  _th = int(np.ceil(thickness/2))
  for af in affinities:
    for i, (x1, y1, x2, y2) in enumerate(af):
      if x1+y1 == 0 or x2+y2 == 0:
        continue
      x1 *= rx
      y1 *= ry
      x2 *= rx
      y2 *= ry
      _x0 = max(min(x1-_th, x2-_th, paf.shape[1]-1), 0)
      _x1 = min(max(x1+_th, x2+_th, 0), paf.shape[1]-1)
      _y0 = max(min(y1-_th, y2-_th, paf.shape[0]-1), 0)
      _y1 = min(max(y1+_th, y2+_th, 0), paf.shape[0]-1)
      _x0 = int(_x0)
      _x1 = int(_x1)
      _y0 = int(_y0)
      _y1 = int(_y1)
      X, Y = np.meshgrid(np.arange(_x0,_x1+1), np.arange(_y0,_y1+1))
      
      vx, vy = x2 - x1, y2 - y1
      l = np.hypot(vx, vy)
      vx /= l
      vy /= l
      wx, wy = vy, -vx
      ux, uy = X - x1, Y - y1
      along = ux * vx + uy * vy
      across = ux * wx + uy * wy
      along = np.logical_and(along > 0, along < l)
      across = np.abs(across) < thickness/2
      mask = np.uint8(np.logical_and(along, across))
      
      paf[_y0:_y1+1,_x0:_x1+1,i*2] += vx * mask
      paf[_y0:_y1+1,_x0:_x1+1,i*2+1] += vy * mask
      npaf[_y0:_y1+1,_x0:_x1+1,i*2] += mask
      npaf[_y0:_y1+1,_x0:_x1+1,i*2+1] += mask
  paf /= np.maximum(npaf, 1)
  return paf

def _to_camel_case(s):
  return ''.join(map(lambda x: x.capitalize(), s.split('_')))

def load_coco_dataset(body_annotation_file, foot_annotation_file=None,
                      keypoint_names=None, limb_names=None,
                      ignore_no_person=True, ignore_no_keypoint=True):
  if keypoint_names is None:
    from labels import POSE_BODY_25_L2 as keypoint_names
  if limb_names is None:
    from labels import POSE_BODY_25_L1 as limb_names

  body_ann = json.load(open(body_annotation_file))
  body_names = body_ann['categories'][0]['keypoints']
  foot_ann = None
  if foot_annotation_file is not None:
    foot_ann = json.load(open(foot_annotation_file))

  # body_25
  body_inds = [keypoint_names.index(_to_camel_case(n).replace('Right','R').replace('Left','L')) \
               for n in body_names]
  foot_inds = [19,20,21,22,23,24]
  limbs = [(keypoint_names.index(s), keypoint_names.index(t)) for s,t in limb_names]

  images = []     # [num_images]
  keypoints = []  # [num_images, num_persons, num_keypoints, 2(x,y)]
  affinities = [] # [num_images, num_persons, num_limbs, 4(x1,y1,x2,y2)]
  K = len(body_ann['images'])
  import sys
  for k, img in enumerate(body_ann['images']):
    sys.stdout.write('\r{}/{}'.format(k+1, K))
    sys.stdout.flush()
    of_image = lambda x: x['image_id'] == img['id']
    bodies = filter(of_image, body_ann['annotations'])
    bodies = {b['id']: b['keypoints'] for b in bodies}
    feet = {}
    if foot_ann is not None:
      feet = filter(of_image, foot_ann['annotations'])
      feet = {f['id']: f['keypoints'] for f in feet}
    persons = set(bodies.keys()).union(feet.keys())
    n_persons = len(persons)
    if ignore_no_person and n_persons == 0:
        continue
    kp = np.zeros([n_persons, len(keypoint_names)-1, 2], np.int32)
    af = np.zeros([n_persons, len(limb_names), 4], np.int32)
    n_keypoints = np.zeros([n_persons], np.int32)
    for i, p in enumerate(persons):
      if p in bodies:
        body = bodies[p]
        v = np.array(body[2::3]) >= 1
        n_keypoints[i] += v.sum()
        kp[i,body_inds,0] = body[0::3] * v
        kp[i,body_inds,1] = body[1::3] * v
      if kp[i,2].sum() > 0 and kp[i,5].sum() > 0: # body_25, Neck
        kp[i,1] = (kp[i,2] + kp[i,5]) // 2
        n_keypoints[i] += 1
      if kp[i,9].sum() > 0 and kp[i,12].sum() > 0: # body_25, MidHip
        kp[i,8] = (kp[i,9] + kp[i,12]) // 2
        n_keypoints[i] += 1
      if p in feet:
        foot = feet[p][-6*3:]
        v = np.array(foot[2::3]) >= 1
        n_keypoints[i] += v.sum()
        kp[i,foot_inds,0] = foot[0::3] * v
        kp[i,foot_inds,1] = foot[1::3] * v
      for j, (s,t) in enumerate(limbs):
        af[i,j,:2] = kp[i,s]
        af[i,j,2:] = kp[i,t]
    if ignore_no_keypoint:
      kp = kp[n_keypoints > 0]
      af = af[n_keypoints > 0]
    if ignore_no_person and kp.shape[0] == 0:
      continue      
    images.append(img['file_name'])
    keypoints.append(kp)
    affinities.append(af)
  return images, keypoints, affinities
