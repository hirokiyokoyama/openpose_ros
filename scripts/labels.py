#!/usr/bin/env python
POSE_COCO_L2 = [
	'nose',
	'neck',
	'right_shoulder',
	'right_elbow',
	'right_wrist',
	'left_shoulder',
	'left_elbow',
	'left_wrist',
	'right_hip',
	'right_knee',
	'right_ankle',
	'left_hip',
	'left_knee',
	'left_ankle',
	'right_eye',
	'left_eye',
	'right_ear',
	'left_ear',
	'other_place'
]

POSE_COCO_L1 = [
	('neck', 'right_hip'),
	('right_hip', 'right_knee'),
	('right_knee', 'right_ankle'),
	('neck', 'left_hip'),
	('left_hip', 'left_knee'),
	('left_knee', 'left_ankle'),
	('neck', 'right_shoulder'),
	('right_shoulder', 'right_elbow'),
	('right_elbow', 'right_wrist'),
	('right_shoulder', 'right_ear'),
	('neck', 'left_shoulder'),
	('left_shoulder', 'left_elbow'),
	('left_elbow', 'left_wrist'),
	('left_shoulder', 'left_ear'),
	('neck', 'nose'),
	('nose', 'right_eye'),
	('nose', 'left_eye'),
	('right_eye', 'right_ear'),
	('left_eye', 'left_ear')
]

HAND = [
	'wrist',
	'thumb_root',
	'thumb_first_joint',
	'thumb_second_joint',
	'thumb_claw',
	'index_first_joint',
	'index_second_joint',
	'index_third_joint',
	'index_claw',
	'middle_first_joint',
	'middle_second_joint',
	'middle_third_joint',
	'middle_claw',
	'third_first_joint',
	'third_second_joint',
	'third_third_joint',
	'third_claw',
	'fourth_first_joint',
	'fourth_second_joint',
	'fourth_third_joint',
	'fourth_claw'
]

POSE_MPI_L2 = [
	'head',
	'neck',
	'right_shoulder',
	'right_elbow',
	'right_wrist',
	'left_shoulder,'
	'left_elbow',
	'left_wrist',
	'right_hip',
	'right_knee',
	'right_ankle',
	'left_hip',
	'left_knee',
	'left_ankle',
	'belly_buttom',
	'other_place'
]

POSE_MPI_L1 = [
	('head','neck'),
	('neck','right_shoulder'),
	('right_shoulder', 'right_elbow'),
	('right_elbow', 'right_wrist'),
	('neck', 'left_shoulder'),
	('left_shoulder', 'left_elbow'),
	('left_elbow', 'left_wrist'),
	('neck', 'belly_buttom'),
	('belly_buttom', 'right_hip'),
	('right_hip', 'right_knee'),
	('right_knee', 'right_ankle'),
	('belly_buttom', 'left_hip'),
	('left_hip', 'left_knee'),
	('left_knee', 'left_ankle')
]
