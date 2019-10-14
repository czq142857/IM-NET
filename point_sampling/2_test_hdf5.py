import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random

class_name = "03001627_chair"
hdf5_path = class_name+"/"+class_name[:8]+"_vox256.hdf5"
voxel_input = h5py.File(hdf5_path, 'r')
voxel_input_voxels = voxel_input["voxels"][:]
voxel_input_points_16 = voxel_input["points_16"][:]
voxel_input_values_16 = voxel_input["values_16"][:]
voxel_input_points_32 = voxel_input["points_32"][:]
voxel_input_values_32 = voxel_input["values_32"][:]
voxel_input_points_64 = voxel_input["points_64"][:]
voxel_input_values_64 = voxel_input["values_64"][:]

if not os.path.exists("tmp"):
	os.makedirs("tmp")

for idx in range(64):
	vox = voxel_input_voxels[idx,:,:,:,0]*255
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_vox_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_vox_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_vox_3.png",img3)

	vox = np.zeros([256,256,256],np.uint8)
	batch_points_int = voxel_input_points_16[idx]
	batch_values = voxel_input_values_16[idx]
	vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_p16_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_p16_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_p16_3.png",img3)
	
	vox = np.zeros([256,256,256],np.uint8)
	batch_points_int = voxel_input_points_32[idx]
	batch_values = voxel_input_values_32[idx]
	vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_p32_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_p32_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_p32_3.png",img3)
	
	vox = np.zeros([256,256,256],np.uint8)
	batch_points_int = voxel_input_points_64[idx]
	batch_values = voxel_input_values_64[idx]
	vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_p64_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_p64_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_p64_3.png",img3)
	