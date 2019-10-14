import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json
from multiprocessing import Process, Queue
import queue
import time
#import mcubes

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

class_name = "03001627_chair"

dim = 64

vox_size_1 = 16
vox_size_2 = 32
vox_size_3 = 64

batch_size_1 = 16*16*16
batch_size_2 = 16*16*16
batch_size_3 = 16*16*16*4




'''
#do not use progressive sampling (center2x2x2 -> 4x4x4 -> 6x6x6 ->...)
#if sample non-center points only for inner(1)-voxels,
#the reconstructed model will have railing patterns.
#since all zero-points are centered at cells,
#the model will expand one-points to a one-planes.
'''
def sample_point_in_cube(block,target_value,halfie):
	halfie2 = halfie*2
	
	for i in range(100):
		x = np.random.randint(halfie2)
		y = np.random.randint(halfie2)
		z = np.random.randint(halfie2)
		if block[x,y,z]==target_value:
			return x,y,z
	
	if block[halfie,halfie,halfie]==target_value:
		return halfie,halfie,halfie
	
	i=1
	ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
	if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
		return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	
	for i in range(2,halfie+1):
		six = [(halfie-i,halfie,halfie),(halfie+i-1,halfie,halfie),(halfie,halfie,halfie-i),(halfie,halfie,halfie+i-1),(halfie,halfie-i,halfie),(halfie,halfie+i-1,halfie)]
		for j in range(6):
			if block[six[j]]==target_value:
				return six[j]
		ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
		if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
			return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	print('hey, error in your code!')
	exit(0)




def get_points_from_vox(q, name_list):
	name_num = len(name_list)
	for idx in range(name_num):
		print(idx,'/',name_num)
		#get voxel models
		try:
			voxel_model_mat = loadmat(name_list[idx][1])
		except:
			print("error in loading")
			exit(-1)
		
		
		
		voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
		voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
		voxel_model_256 = np.zeros([256,256,256],np.uint8)
		for i in range(16):
			for j in range(16):
				for k in range(16):
					voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = voxel_model_b[voxel_model_bi[i,j,k]]
		#add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
		voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)
		
		
		#vertices, triangles = mcubes.marching_cubes(voxel_model_256, 0.5)
		#mcubes.export_mesh(vertices, triangles, "samples/"+name_list[idx][1][-10:-4]+"_origin.dae", str(idx))
		
		
		#carve the voxels from side views:
		#top direction = Y(j) positive direction
		dim_voxel = 256
		top_view = np.max(voxel_model_256, axis=1)
		left_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
		left_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
		front_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
		front_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
		
		for j in range(dim_voxel):
			for k in range(dim_voxel):
				occupied = False
				for i in range(dim_voxel):
					if voxel_model_256[i,j,k]>0:
						if not occupied:
							occupied = True
							left_min[j,k] = i
						left_max[j,k] = i
		
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				occupied = False
				for k in range(dim_voxel):
					if voxel_model_256[i,j,k]>0:
						if not occupied:
							occupied = True
							front_min[i,j] = k
						front_max[i,j] = k
		
		for i in range(dim_voxel):
			for k in range(dim_voxel):
				if top_view[i,k]>0:
					fill_flag = False
					for j in range(dim_voxel-1,-1,-1):
						if voxel_model_256[i,j,k]>0:
							fill_flag = True
						else:
							if left_min[j,k]<i and left_max[j,k]>i and front_min[i,j]<k and front_max[i,j]>k:
								if fill_flag:
									voxel_model_256[i,j,k]=1
							else:
								fill_flag = False
		
		#vertices, triangles = mcubes.marching_cubes(voxel_model_256, 0.5)
		#mcubes.export_mesh(vertices, triangles, "samples/"+name_list[idx][1][-10:-4]+"_alt.dae", str(idx))
		
		
		
		
		
		#compress model 256 -> 64
		dim_voxel = 64
		voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_temp[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		#write voxel
		sample_voxels = np.reshape(voxel_model_temp, (dim_voxel,dim_voxel,dim_voxel,1))
		
		#sample points near surface
		batch_size = batch_size_3
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		for j in temp_range:
			if (batch_size_counter>=batch_size): break
			for i in temp_range:
				if (batch_size_counter>=batch_size): break
				for k in temp_range:
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
						si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
						sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
						voxel_model_temp_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			print("64-- batch_size exceeded!")
			exceed_64_flag = 1
		else:
			exceed_64_flag = 0
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_temp_flag[i,j,k] != 1: break
				si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		sample_points_64 = sample_points
		sample_values_64 = sample_values
		
		
		
		
		
		#compress model 256 -> 32
		dim_voxel = 32
		voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_temp[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		#sample points near surface
		batch_size = batch_size_2
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		for j in temp_range:
			if (batch_size_counter>=batch_size): break
			for i in temp_range:
				if (batch_size_counter>=batch_size): break
				for k in temp_range:
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
						si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
						sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
						voxel_model_temp_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			print("32-- batch_size exceeded!")
			exceed_32_flag = 1
		else:
			exceed_32_flag = 0
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_temp_flag[i,j,k] != 1: break
				si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		sample_points_32 = sample_points
		sample_values_32 = sample_values
		
		
		
		
		
		#compress model 256 -> 16
		dim_voxel = 16
		voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_temp[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		#sample points near surface
		batch_size = batch_size_1
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
					sample_points[batch_size_counter,0] = si+i*multiplier
					sample_points[batch_size_counter,1] = sj+j*multiplier
					sample_points[batch_size_counter,2] = sk+k*multiplier
					sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
					batch_size_counter +=1
		if (batch_size_counter!=batch_size):
			print("batch_size_counter!=batch_size")
		
		sample_points_16 = sample_points
		sample_values_16 = sample_values
		
		q.put([name_list[idx][0],exceed_64_flag,exceed_32_flag,sample_points_64,sample_values_64,sample_points_32,sample_values_32,sample_points_16,sample_values_16,sample_voxels])




def list_image(root, exts):
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






if __name__ == '__main__':
	print(class_name)
	if not os.path.exists(class_name):
		os.makedirs(class_name)

	#dir of voxel models
	voxel_input = "/local-scratch/zhiqinc/shapenet_hsp/modelBlockedVoxels256/"+class_name[:8]+"/"

	#name of output file
	hdf5_path = class_name+'/'+class_name[:8]+'_vox256.hdf5'

	#obj_list
	fout = open(class_name+'/'+class_name[:8]+'_vox256.txt','w',newline='')
	
	#record statistics
	fstatistics = open(class_name+'/statistics.txt','w',newline='')
	exceed_32 = 0
	exceed_64 = 0

	image_list = list_image(voxel_input, ['.mat'])
	name_list = []
	for i in range(len(image_list)):
		imagine=image_list[i][0]
		name_list.append(imagine[0:-4])
	name_list = sorted(name_list)
	name_num = len(name_list)

	for i in range(name_num):
		fout.write(name_list[i]+"\n")
	fout.close()
	
	#prepare list of names
	num_of_process = 12
	list_of_list_of_names = []
	for i in range(num_of_process):
		list_of_names = []
		for j in range(i,name_num,num_of_process):
			list_of_names.append([j, voxel_input+name_list[j]+".mat"])
		list_of_list_of_names.append(list_of_names)
	
	#map processes
	q = Queue()
	workers = [Process(target=get_points_from_vox, args = (q, list_of_names)) for list_of_names in list_of_list_of_names]

	for p in workers:
		p.start()


	#reduce process
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("voxels", [name_num,dim,dim,dim,1], np.uint8)
	hdf5_file.create_dataset("points_16", [name_num,batch_size_1,3], np.uint8)
	hdf5_file.create_dataset("values_16", [name_num,batch_size_1,1], np.uint8)
	hdf5_file.create_dataset("points_32", [name_num,batch_size_2,3], np.uint8)
	hdf5_file.create_dataset("values_32", [name_num,batch_size_2,1], np.uint8)
	hdf5_file.create_dataset("points_64", [name_num,batch_size_3,3], np.uint8)
	hdf5_file.create_dataset("values_64", [name_num,batch_size_3,1], np.uint8)



	while True:
		item_flag = True
		try:
			idx,exceed_64_flag,exceed_32_flag,sample_points_64,sample_values_64,sample_points_32,sample_values_32,sample_points_16,sample_values_16,sample_voxels = q.get(True, 1.0)
		except queue.Empty:
			item_flag = False
		
		if item_flag:
			#process result
			exceed_32+=exceed_32_flag
			exceed_64+=exceed_64_flag
			hdf5_file["points_64"][idx,:,:] = sample_points_64
			hdf5_file["values_64"][idx,:,:] = sample_values_64
			hdf5_file["points_32"][idx,:,:] = sample_points_32
			hdf5_file["values_32"][idx,:,:] = sample_values_32
			hdf5_file["points_16"][idx,:,:] = sample_points_16
			hdf5_file["values_16"][idx,:,:] = sample_values_16
			hdf5_file["voxels"][idx,:,:,:,:] = sample_voxels
		
		allExited = True
		for p in workers:
			if p.exitcode is None:
				allExited = False
				break
		if allExited and q.empty():
			break




	fstatistics.write("total: "+str(name_num)+"\n")
	fstatistics.write("exceed_32: "+str(exceed_32)+"\n")
	fstatistics.write("exceed_32_ratio: "+str(float(exceed_32)/name_num)+"\n")
	fstatistics.write("exceed_64: "+str(exceed_64)+"\n")
	fstatistics.write("exceed_64_ratio: "+str(float(exceed_64)/name_num)+"\n")
	
	fstatistics.close()
	hdf5_file.close()
	print("finished")


