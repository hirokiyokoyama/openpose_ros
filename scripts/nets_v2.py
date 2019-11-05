import rospy
import tensorflow as tf
import numpy as np
import scipy
from scipy import optimize
layers = tf.keras.layers

class OpenPoseCPM(tf.keras.Model):
  def __init__(self, dtype=tf.float32):
    super(OpenPoseCPM, self).__init__(dtype=dtype)
    self._layers = [
        layers.Conv2D(64, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.Conv2D(64, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.MaxPool2D(2, padding='SAME', dtype=dtype),
        layers.Conv2D(128, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.Conv2D(128, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.MaxPool2D(2, padding='SAME', dtype=dtype),
        layers.Conv2D(256, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.Conv2D(256, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.Conv2D(256, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.Conv2D(256, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.MaxPool2D(2, padding='SAME', dtype=dtype),
        layers.Conv2D(512, [3,3], 1, padding='SAME', dtype=dtype),
        layers.ReLU(dtype=dtype),
        layers.Conv2D(512, [3,3], 1, padding='SAME', dtype=dtype),
        layers.PReLU(shared_axes=[1,2], dtype=dtype),
        layers.Conv2D(256, [3,3], 1, padding='SAME', dtype=dtype),
        layers.PReLU(shared_axes=[1,2], dtype=dtype),
        layers.Conv2D(128, [3,3], 1, padding='SAME', dtype=dtype),
        layers.PReLU(shared_axes=[1,2], dtype=dtype),
    ]

  def call(self, x):
    for l in self._layers:
      x = l(x)
    return x

class OpenPoseStage(tf.keras.Model):
  def __init__(self, c, c1=96, c2=256, dtype=tf.float32):
    super(OpenPoseStage, self).__init__(dtype=dtype)

    self._blocks = []
    for i in range(5):
      self._blocks.append([
          layers.Conv2D(c1, [3,3], 1, padding='SAME', dtype=dtype),
          layers.PReLU(shared_axes=[1,2], dtype=dtype),
          layers.Conv2D(c1, [3,3], 1, padding='SAME', dtype=dtype),
          layers.PReLU(shared_axes=[1,2], dtype=dtype),
          layers.Conv2D(c1, [3,3], 1, padding='SAME', dtype=dtype),
          layers.PReLU(shared_axes=[1,2], dtype=dtype),
      ])
      
    self._output_layers = [
        layers.Conv2D(c2, [1,1], 1, padding='SAME', dtype=dtype),
        layers.PReLU(shared_axes=[1,2], dtype=dtype),
        layers.Conv2D(c, [1,1], 1, padding='SAME', dtype=dtype),
    ]
    
  def call(self, x):
    for conv1, relu1, conv2, relu2, conv3, relu3 in self._blocks:
      x1 = relu1(conv1(x))
      x2 = relu2(conv2(x1))
      x3 = relu3(conv3(x2))
      x = tf.concat([x1, x2, x3], -1)
    for l in self._output_layers:
      x = l(x)
    return x
  
class OpenPose(tf.keras.Model):
  def __init__(self, num_parts=26, num_limbs=26, part_stages=1, limb_stages=3, dtype=tf.float32):
    super(OpenPose, self).__init__(dtype=dtype)

    self._cpm = OpenPoseCPM(dtype=dtype)
    
    self._limb_stages = [OpenPoseStage(num_limbs*2, 96, 256, dtype=dtype)]
    for i in range(limb_stages):
      self._limb_stages.append(OpenPoseStage(num_limbs*2, 128, 512, dtype=dtype))
      
    self._part_stages = [OpenPoseStage(num_parts, 96, 256, dtype=dtype)]
    for i in range(part_stages):
      self._part_stages.append(OpenPoseStage(num_parts, 128, 512, dtype=dtype))
      
  def call(self, x):
    cpm = self._cpm(x)
    
    limb = self._limb_stages[0](cpm)
    for stage in self._limb_stages[1:]:
      limb = tf.concat([cpm, limb], -1)
      limb = stage(limb)
      
    part = tf.concat([cpm, limb], -1)
    part = self._part_stages[0](part)
    for stage in self._part_stages[1:]:
      part = tf.concat([cpm, part, limb], -1)
      part = stage(part)
    
    return part, limb

def pose_net_body_25(dtype=tf.float32):
  return OpenPose(num_parts=26, num_limbs=26, dtype=dtype)

@tf.function
def non_maximum_suppression(heat_map, threshold=tf.constant(0.5)):
  heat_map_max = tf.nn.max_pool2d(
    heat_map, ksize=[3,3], strides=1, padding='SAME')
  # (num, 4(NHWC))
  inds = tf.where(
    tf.logical_and(heat_map > threshold,
                   tf.equal(heat_map_max, heat_map)))
  #return tf.concat(values=[inds[:,0:1], inds[:, 1:3] * 8, inds[:,3:]], axis=1)
  return inds

def connect_parts(affinity, keypoints, limbs, line_division=10, threshold=0.2):
  persons = [{c: id} for id, (_,_,c) in enumerate(keypoints)]
  for k, (p, q) in enumerate(limbs):
    is_p = keypoints[:,2] == p
    is_q = keypoints[:,2] == q
    p_inds = np.where(is_p)[0]
    q_inds = np.where(is_q)[0]
    q_mesh, p_mesh = np.meshgrid(np.where(is_q), np.where(is_p))
    Px = keypoints[p_mesh, 1]
    Py = keypoints[p_mesh, 0]
    Qx = keypoints[q_mesh, 1]
    Qy = keypoints[q_mesh, 0]
    Dx = Qx - Px
    Dy = Qy - Py
    norm = np.sqrt(Dx**2 + Dy**2)
    piled = norm==0.
    norm[piled] = 1
    Dx = Dx/norm
    Dy = Dy/norm
    Lx = np.zeros_like(Dx)
    Ly = np.zeros_like(Dy)
    for u in np.linspace(0,1,line_division):
      Rx = np.int32((1-u) * Px + u * Qx)
      Ry = np.int32((1-u) * Py + u * Qy)
      Lx += affinity[Ry,Rx,k*2]
      Ly += affinity[Ry,Rx,k*2+1]
    C = (Dx*Lx + Dy*Ly)/line_division
    C[piled] = threshold
    # rospy.loginfo('norm==0: {}'.format(np.where(norm==0)))
    # if k==10:
    #   rospy.loginfo(C)
    # I, J = scipy.optimize.linear_sum_assignment(-C)
    I, J = optimize.linear_sum_assignment(-C)
    for i, j in zip(I, J):
      if C[i,j] < threshold:
        continue
      i = p_inds[i]
      j = q_inds[j]
      matched = list(filter(lambda person: i in person.values(), persons))
      matched.extend(filter(lambda person: j in person.values(), persons))
      if len(matched) > 1:
        # rospy.loginfo('{}->{}: {} entries will be merged.'.format(i, j, len(matched)))
        merged = {}
        for person in matched:
          merged.update(person)
          if person in persons:
            persons.remove(person)
        # rospy.loginfo('{} -> {}'.format(matched, merged))
        persons.append(merged)
  #return [{k:keypoints[v,:2] for k,v in p.iteritems()} for p in persons]
  return persons
