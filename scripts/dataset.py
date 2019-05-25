import cv2
import json
import os
import numpy as np

def generate_cm(img_shape, keypoints, sigma=8.):
  cm = np.zeros(img_shape+(keypoints.shape[1]+1,))

  # prepare gaussian patch
  patch_size = int(np.ceil(3*sigma))
  space = np.linspace(-3*sigma, 3*sigma, patch_size * 2 + 1)
  X, Y = np.meshgrid(space, space)
  gaussian_patch = np.exp(-(X**2+Y**2)/(2*sigma**2))
    
  for kp in keypoints:
    for i, (x, y) in enumerate(kp):
      if x+y == 0:
        continue
      _x0 = min(max(x-patch_size, 0), cm.shape[1]-1)
      _x1 = min(max(x+patch_size, 0), cm.shape[1]-1)
      __x0 = max(_x0 - (x-patch_size), 0)
      __x1 = __x0 + _x1 - _x0
      _y0 = min(max(y-patch_size, 0), cm.shape[0]-1)
      _y1 = min(max(y+patch_size, 0), cm.shape[0]-1)
      __y0 = max(_y0 - (y-patch_size), 0)
      __y1 = __y0 + _y1 - _y0
      cm[_y0:_y1+1,_x0:_x1+1,i] = np.maximum(
        cm[_y0:_y1+1,_x0:_x1+1,i], gaussian_patch[__y0:__y1+1,__x0:__x1+1])
  cm[:,:,-1] = 1. - cm[:,:,:-1].max(axis=2)
  return cm

def generate_paf(img_shape, affinities, thickness=8.):
  paf = np.zeros(img_shape+(affinities.shape[1]*2,))
  npaf = np.zeros(paf.shape, np.int32)
  _th = int(np.ceil(thickness/2))
  for af in affinities:
    for i, (x1, y1, x2, y2) in enumerate(af):
      if x1+y1 == 0 or x2+y2 == 0:
        continue
      _x0 = max(min(x1-_th, x2-_th, paf.shape[1]-1), 0)
      _x1 = min(max(x1+_th, x2+_th, 0), paf.shape[1]-1)
      _y0 = max(min(y1-_th, y2-_th, paf.shape[0]-1), 0)
      _y1 = min(max(y1+_th, y2+_th, 0), paf.shape[0]-1)
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
                      keypoint_names=None, limb_names=None):
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
  for img in body_ann['images']:
    of_image = lambda x: x['image_id'] == img['id']
    bodies = filter(of_image, body_ann['annotations'])
    bodies = {b['id']: b['keypoints'] for b in bodies}
    feet = {}
    if foot_ann is not None:
      feet = filter(of_image, foot_ann['annotations'])
      feet = {f['id']: f['keypoints'] for f in feet}
    persons = set(bodies.keys()).union(feet.keys())
    n_persons = len(persons)
    if n_persons == 0:
        continue
    kp = np.zeros([n_persons, len(keypoint_names)-1, 2], np.int32)
    af = np.zeros([n_persons, len(limb_names), 4], np.int32)
    for i, p in enumerate(persons):
      if p in bodies:
        body = bodies[p]
        v = np.array(body[2::3]) >= 1
        kp[i,body_inds,0] = body[0::3] * v
        kp[i,body_inds,1] = body[1::3] * v
      if kp[i,2].sum() > 0 and kp[i,5].sum() > 0: # body_25, Neck
        kp[i,1] = (kp[i,2] + kp[i,5]) // 2
      if kp[i,9].sum() > 0 and kp[i,12].sum() > 0: # body_25, MidHip
        kp[i,8] = (kp[i,9] + kp[i,12]) // 2
      if p in feet:
        foot = feet[p]
        v = np.array(foot[2::3]) >= 1
        kp[i,foot_inds,0] = foot[0::3] * v
        kp[i,foot_inds,1] = foot[1::3] * v
      for j, (s,t) in enumerate(limbs):
        af[i,j,:2] = kp[i,s]
        af[i,j,2:] = kp[i,t]
    images.append(img['file_name'])
    keypoints.append(kp)
    affinities.append(af)
  return images, keypoints, affinities
  
if __name__=='__main__':
  from labels import POSE_BODY_25_L2, POSE_BODY_25_L1
  FOOT_FILE = '/media/psf/Home/Documents/coco/annotations/person_keypoints_val2017_foot_v1.json'
  BODY_FILE = '/media/psf/Home/Documents/coco/annotations/person_keypoints_val2017.json'
  IMAGE_DIR = '/media/psf/Home/Documents/coco/images/val2017'
  images, keypoints, affinities = load_coco_dataset(
    BODY_FILE, FOOT_FILE,
    keypoint_names=POSE_BODY_25_L2,
    limb_names=POSE_BODY_25_L1)
  
  sigma = 8.
  thickness = 8.
  j = 0
  while True:
    imgfile = os.path.join(IMAGE_DIR, images[j])
    img = cv2.imread(imgfile)

    n_persons = len(keypoints[j])

    cm = generate_cm(img.shape[:2], keypoints[j], sigma=8.)
    paf = generate_paf(img.shape[:2], affinities[j], thickness=thickness)

    cv2.imshow('ConfidenceMap', np.uint8((img + cm[:,:,0:3]*255)/2))
    cv2.imshow('Background', np.uint8(cm[:,:,-1]*255))
    pafimg = np.stack([np.arctan2(-paf[:,:,0], paf[:,:,1])/(2*np.pi)*180+90,
                       np.ones(paf.shape[:2])*255,
                       np.hypot(paf[:,:,0], paf[:,:,1])*255], 2)
    pafimg = cv2.cvtColor(np.uint8(pafimg), cv2.COLOR_HSV2BGR)
    cv2.imshow('PartAffinityField', np.uint8(img*.5 + pafimg*.5))
    
    key = cv2.waitKey()
    if key == ord(' '):
      break
    j += 1
