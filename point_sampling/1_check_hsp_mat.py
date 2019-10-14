import numpy as np
import os
import h5py
from scipy.io import loadmat
import random
import json

class_name_list_all = [
"02691156_airplane",
"02828884_bench",
"02933112_cabinet",
"02958343_car",
"03001627_chair",
"03211117_display",
"03636649_lamp",
"03691459_speaker",
"04090263_rifle",
"04256520_couch",
"04379243_table",
"04401088_phone",
"04530566_vessel",
]

class_name_list = [
"02691156_airplane",
"02828884_bench",
"02933112_cabinet",
"02958343_car",
"03001627_chair",
"03211117_display",
"03636649_lamp",
"03691459_speaker",
"04090263_rifle",
"04256520_couch",
"04379243_table",
"04401088_phone",
"04530566_vessel",
]


def list_image(root, recursive, exts):
	image_list = []
	cat = {}
	for path, subdirs, files in os.walk(root):
		for fname in files:
			fpath = os.path.join(path, fname)
			suffix = os.path.splitext(fname)[1].lower()
			if os.path.isfile(fpath) and (suffix in exts):
				if path not in cat:
					cat[path] = len(cat)
				image_list.append((os.path.relpath(fpath, root), cat[path]))
	return image_list


for kkk in range(len(class_name_list)):
	class_name = class_name_list[kkk][:8]
	print(class_name_list[kkk])
	voxel_input = "/local-scratch/zhiqinc/shapenet_hsp/modelBlockedVoxels256/"+class_name+"/"
	image_list = list_image(voxel_input, True, ['.mat'])
	name_list = []
	for i in range(len(image_list)):
		imagine=image_list[i][0]
		name_list.append(imagine[0:-4])
	name_list = sorted(name_list)
	name_num = len(name_list)
	for idx in range(name_num):
		name_list_idx = name_list[idx]
		proper_name = "shape_data/"+class_name+"/"+name_list_idx
		try:
			voxel_model_mat = loadmat(voxel_input+name_list_idx+".mat")
		except:
			print("error in loading")
			print(voxel_input+name_list_idx+".mat")
