import numpy as np
import cv2
import os
import h5py
import random


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

is_train_set = True
voxel_input_dir = "/home/zhiqinc/zhiqinc/shapenet_hsp/HSPvoxels256/"
img_input_dir = "/home/zhiqinc/zhiqinc/shapenet_hsp/HSPvoxels/"

for kkk in range(len(class_name_list)):
	if not os.path.exists(class_name_list[kkk]):
		os.makedirs(class_name_list[kkk])
	class_name = class_name_list[kkk][:8]
	# step 1
	# count number of shapes
	# make a list of all shape names
	num_shapes = 0
	if is_train_set:
		output_shape_name_list = open(class_name_list[kkk]+'/'+class_name+"_vox256_img_train.txt", 'w')
	else:
		output_shape_name_list = open(class_name_list[kkk]+'/'+class_name+"_vox256_img_test.txt", 'w')

	input_txt_dir = voxel_input_dir+'/'+class_name_list[kkk]+'/'+class_name+'_vox256.txt'
	input_txt = open(input_txt_dir, 'r')
	# this list is already sorted
	input_list = input_txt.readlines()
	input_txt.close()
	input_len = len(input_list)
	if is_train_set:
		start_len = 0
		target_len = int(input_len*0.8)
		num_shapes += target_len
	else:
		start_len = int(input_len*0.8)
		target_len = input_len-start_len
		num_shapes += target_len
	for i in range(target_len):
		output_shape_name_list.write(class_name+'/'+input_list[start_len+i].strip()+'\n')
	output_shape_name_list.close()


	# step 2
	# write voxels
	# write images

	num_view = 24
	view_size = 137
	vox_size = 64
	vox_size_1 = 16
	vox_size_2 = 32
	vox_size_3 = 64
	batch_size_1 = 16*16*16
	batch_size_2 = 16*16*16
	batch_size_3 = 16*16*16*4

	if is_train_set:
		hdf5_file = h5py.File(class_name_list[kkk]+'/'+class_name+"_vox256_img_train.hdf5", 'w')
	else:
		hdf5_file = h5py.File(class_name_list[kkk]+'/'+class_name+"_vox256_img_test.hdf5", 'w')
	hdf5_file.create_dataset("pixels", [num_shapes,num_view,view_size,view_size], np.uint8, compression=9)
	hdf5_file.create_dataset("voxels", [num_shapes,vox_size,vox_size,vox_size,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_16", [num_shapes,batch_size_1,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_16", [num_shapes,batch_size_1,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_32", [num_shapes,batch_size_2,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_32", [num_shapes,batch_size_2,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_64", [num_shapes,batch_size_3,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_64", [num_shapes,batch_size_3,1], np.uint8, compression=9)

	counter = 0
	input_txt_dir = voxel_input_dir+'/'+class_name_list[kkk]+'/'+class_name+'_vox256.txt'
	input_txt = open(input_txt_dir, 'r')
	# this list is already sorted
	input_list = input_txt.readlines()
	input_txt.close()
	input_len = len(input_list)
	shape_name_list = []
	if is_train_set:
		start_len = 0
		target_len = int(input_len*0.8)
	else:
		start_len = int(input_len*0.8)
		target_len = input_len-start_len
	for i in range(target_len):
		shape_name_list.append(input_list[start_len+i].strip())
	
	voxel_hdf5_dir1 = voxel_input_dir+'/'+class_name_list[kkk]+'/'+class_name+'_vox256.hdf5'
	voxel_hdf5_file1 = h5py.File(voxel_hdf5_dir1, 'r')
	voxel_hdf5_voxels = voxel_hdf5_file1['voxels'][:]
	voxel_hdf5_points_16 = voxel_hdf5_file1['points_16'][:]
	voxel_hdf5_values_16 = voxel_hdf5_file1['values_16'][:]
	voxel_hdf5_points_32 = voxel_hdf5_file1['points_32'][:]
	voxel_hdf5_values_32 = voxel_hdf5_file1['values_32'][:]
	voxel_hdf5_points_64 = voxel_hdf5_file1['points_64'][:]
	voxel_hdf5_values_64 = voxel_hdf5_file1['values_64'][:]
	voxel_hdf5_file1.close()

	image_hdf5_dir = img_input_dir+class_name_list[kkk]+'/'+class_name+'_img.hdf5'
	image_hdf5_file = h5py.File(image_hdf5_dir, 'r')
	image_hdf5_pixels = image_hdf5_file['pixels'][:]
	image_hdf5_file.close()

	print(counter,num_shapes)

	hdf5_file["pixels"][counter:counter+target_len] = image_hdf5_pixels[start_len:start_len+target_len]
	hdf5_file["voxels"][counter:counter+target_len] = voxel_hdf5_voxels[start_len:start_len+target_len]
	hdf5_file["points_16"][counter:counter+target_len] = voxel_hdf5_points_16[start_len:start_len+target_len]
	hdf5_file["values_16"][counter:counter+target_len] = voxel_hdf5_values_16[start_len:start_len+target_len]
	hdf5_file["points_32"][counter:counter+target_len] = voxel_hdf5_points_32[start_len:start_len+target_len]
	hdf5_file["values_32"][counter:counter+target_len] = voxel_hdf5_values_32[start_len:start_len+target_len]
	hdf5_file["points_64"][counter:counter+target_len] = voxel_hdf5_points_64[start_len:start_len+target_len]
	hdf5_file["values_64"][counter:counter+target_len] = voxel_hdf5_values_64[start_len:start_len+target_len]
	
	counter += target_len

	assert(counter==num_shapes)
	hdf5_file.close()


