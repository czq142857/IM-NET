import numpy as np
import os
import h5py
from scipy.io import loadmat
import random
import json
import cv2

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
	##a lot of customized dirs
	#class number
	class_name = class_name_list[kkk][:8]
	print(class_name_list[kkk])
	#dir of voxel models
	voxel_input = "/local-scratch/zhiqinc/shapenet_hsp/modelBlockedVoxels256/"+class_name+"/"
	image_input = "/home/zhiqinc/zhiqinc/shapenet_atlas/ShapeNetRendering/"+class_name+"/"

	if not os.path.exists(class_name_list[kkk]):
		os.makedirs(class_name_list[kkk])

	#name of output file
	hdf5_path = class_name_list[kkk]+'/'+class_name+'_img.hdf5'

	num_view = 24
	view_size = 137


	image_list = list_image(voxel_input, True, ['.mat'])
	name_list = []
	for i in range(len(image_list)):
		imagine=image_list[i][0]
		name_list.append(imagine[0:-4])
	name_list = sorted(name_list)
	name_num = len(name_list)
	#name_list contains all obj names
	
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("pixels", [name_num,num_view,view_size,view_size], np.uint8, compression=9)

	for idx in range(name_num):
		print(idx)
		
		#get voxel models
		name_list_idx = name_list[idx]
		
		#get rendered views
		for t in range(num_view):
			if t<10:
				image_name = "0"+str(t)+".png"
			else:
				image_name = str(t)+".png"
			img_add = image_input+name_list_idx+'/rendering/'+image_name
			img = cv2.imread(img_add, cv2.IMREAD_UNCHANGED)
			imgo = img[:,:,:3]
			imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
			imga = (img[:,:,3])/255.0
			img = imgo*imga + 255*(1-imga)
			img = np.round(img).astype(np.uint8)
			hdf5_file["pixels"][idx,t,:,:] = img

	hdf5_file.close()
	print("finished")


