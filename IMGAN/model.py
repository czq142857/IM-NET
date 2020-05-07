import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py
import cv2
import mcubes

from ops import *

class IMAE(object):
	def __init__(self, sess, real_size, batch_size_input, is_training = False, z_dim=256, ef_dim=32, gf_dim=128, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.real_size = real_size #output point-value voxel grid size in training
		self.batch_size_input = batch_size_input #training batch size (virtual, batch_size is the real batch_size)
		
		self.batch_size = 32*32*32 #adjust batch_size according to gpu memory size in training
		if self.batch_size_input<self.batch_size:
			self.batch_size = self.batch_size_input
		
		self.input_size = 64 #input voxel grid size

		self.z_dim = z_dim
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim

		self.dataset_name = dataset_name
		self.dataset_load = dataset_name + '_train'
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		#if not is_training:
		#	self.dataset_load = dataset_name + '_test'

		if os.path.exists(self.data_dir+'/'+self.dataset_load+'.hdf5'):
			self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_load+'.hdf5', 'r')
			self.data_points_int = self.data_dict['points_'+str(self.real_size)][:]
			self.data_points = (self.data_points_int.astype(np.float32)+0.5)/256-0.5
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_voxels = self.data_dict['voxels'][:]
			if self.batch_size_input!=self.data_points.shape[1]:
				print("error: batch_size!=data_points.shape")
				exit(0)
			if self.input_size!=self.data_voxels.shape[1]:
				print("error: input_size!=data_voxels.shape")
				exit(0)
		else:
			if is_training:
				print("error: cannot load "+self.data_dir+'/'+self.dataset_load+'.hdf5')
				exit(0)
			else:
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_load+'.hdf5')
		
		if not is_training:
			#keep everything a power of 2
			self.cell_grid_size = 4
			self.frame_grid_size = 64
			self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
			self.batch_size = 16*16*16*4*4 #adjust batch_size according to gpu memory size in testing
			
			#get coords
			dimc = self.cell_grid_size
			dimf = self.frame_grid_size
			self.cell_x = np.zeros([dimc,dimc,dimc],np.int32)
			self.cell_y = np.zeros([dimc,dimc,dimc],np.int32)
			self.cell_z = np.zeros([dimc,dimc,dimc],np.int32)
			self.cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
			self.frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
			self.frame_x = np.zeros([dimf,dimf,dimf],np.int32)
			self.frame_y = np.zeros([dimf,dimf,dimf],np.int32)
			self.frame_z = np.zeros([dimf,dimf,dimf],np.int32)
			for i in range(dimc):
				for j in range(dimc):
					for k in range(dimc):
						self.cell_x[i,j,k] = i
						self.cell_y[i,j,k] = j
						self.cell_z[i,j,k] = k
			for i in range(dimf):
				for j in range(dimf):
					for k in range(dimf):
						self.cell_coords[i,j,k,:,:,:,0] = self.cell_x+i*dimc
						self.cell_coords[i,j,k,:,:,:,1] = self.cell_y+j*dimc
						self.cell_coords[i,j,k,:,:,:,2] = self.cell_z+k*dimc
						self.frame_coords[i,j,k,0] = i
						self.frame_coords[i,j,k,1] = j
						self.frame_coords[i,j,k,2] = k
						self.frame_x[i,j,k] = i
						self.frame_y[i,j,k] = j
						self.frame_z[i,j,k] = k
			self.cell_coords = (self.cell_coords+0.5)/self.real_size-0.5
			self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
			self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
			self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
			self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
			self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
			self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
			self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
			self.frame_coords = (self.frame_coords+0.5)/dimf-0.5
			self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])
			self.sampling_threshold = 0.5 #final marching cubes threshold
		
		self.build_model()

	def build_model(self):
		self.vox3d = tf.placeholder(shape=[1,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32)
		self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32)
		self.point_coord = tf.placeholder(shape=[None,3], dtype=tf.float32)
		self.point_value = tf.placeholder(shape=[None,1], dtype=tf.float32)
		
		self.E = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.G = self.generator(self.point_coord, self.E, phase_train=True, reuse=False)
		self.sE = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.sG = self.generator(self.point_coord, self.sE, phase_train=False, reuse=True)
		self.zG = self.generator(self.point_coord, self.z_vector, phase_train=False, reuse=True)
		
		self.loss = tf.reduce_mean(tf.square(self.point_value - self.G))
		
		self.saver = tf.train.Saver(max_to_keep=10)
		
		
	def generator(self, points, z, phase_train=True, reuse=False):
		batch_size = tf.shape(points)[0]
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			zs = tf.tile(z, [batch_size,1])
			pointz = tf.concat([points,zs],1)
			
			h1 = lrelu(linear(pointz, self.gf_dim*8, 'h1_lin'))
			h2 = lrelu(linear(h1, self.gf_dim*8, 'h2_lin'))
			h3 = lrelu(linear(h2, self.gf_dim*8, 'h3_lin'))
			h4 = lrelu(linear(h3, self.gf_dim*4, 'h4_lin'))
			h5 = lrelu(linear(h4, self.gf_dim*2, 'h5_lin'))
			h6 = lrelu(linear(h5, self.gf_dim, 'h6_lin'))
			h7 = linear(h6, 1, 'h7_lin')
			h7 = tf.maximum(tf.minimum(h7, 1), 0)
			#use this leaky activation function instead if you face convergence issues
			#h7 = tf.maximum(tf.minimum(h7, h7*0.01+0.99), h7*0.01)
			
			return tf.reshape(h7, [batch_size,1])
	
	def encoder(self, inputs, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			d_1 = conv3d(inputs, shape=[4, 4, 4, 1, self.ef_dim], strides=[1,2,2,2,1], scope='conv_1')
			d_1 = lrelu(instance_norm(d_1, phase_train))

			d_2 = conv3d(d_1, shape=[4, 4, 4, self.ef_dim, self.ef_dim*2], strides=[1,2,2,2,1], scope='conv_2')
			d_2 = lrelu(instance_norm(d_2, phase_train))
			
			d_3 = conv3d(d_2, shape=[4, 4, 4, self.ef_dim*2, self.ef_dim*4], strides=[1,2,2,2,1], scope='conv_3')
			d_3 = lrelu(instance_norm(d_3, phase_train))

			d_4 = conv3d(d_3, shape=[4, 4, 4, self.ef_dim*4, self.ef_dim*8], strides=[1,2,2,2,1], scope='conv_4')
			d_4 = lrelu(instance_norm(d_4, phase_train))

			d_5 = conv3d(d_4, shape=[4, 4, 4, self.ef_dim*8, self.z_dim], strides=[1,1,1,1,1], scope='conv_5', padding="VALID")
			d_5 = tf.nn.sigmoid(d_5)
		
			return tf.reshape(d_5,[1,self.z_dim])
	
	def train(self, config):
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		self.sess.run(tf.global_variables_initializer())
		
		batch_idxs = len(self.data_points)
		batch_index_list = np.arange(batch_idxs)
		batch_num = int(self.batch_size_input/self.batch_size)
		if self.batch_size_input%self.batch_size != 0:
			print("batch_size_input % batch_size != 0")
			exit(0)
		
		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in range(counter, config.epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(0, batch_idxs):
				for minib in range(batch_num):
					dxb = batch_index_list[idx]
					batch_voxels = self.data_voxels[dxb:dxb+1]
					batch_points = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
					batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
					
					# Update AE network
					_, errAE = self.sess.run([ae_optim, self.loss],
						feed_dict={
							self.vox3d: batch_voxels,
							self.point_coord: batch_points,
							self.point_value: batch_values,
						})
					avg_loss += errAE
					avg_num += 1
					if (idx%1024 == 0):
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, avgloss: %.8f" % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errAE, avg_loss/avg_num))

				if idx==batch_idxs-1:
					model_float = np.zeros([256,256,256],np.float32)
					real_model_float = np.zeros([256,256,256],np.float32)
					for minib in range(batch_num):
						dxb = batch_index_list[idx]
						batch_voxels = self.data_voxels[dxb:dxb+1]
						batch_points_int = self.data_points_int[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
						batch_points = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
						batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
						
						model_out = self.sess.run(self.sG,
							feed_dict={
								self.vox3d: batch_voxels,
								self.point_coord: batch_points,
							})
						model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(model_out, [self.batch_size])
						real_model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [self.batch_size])
					img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
					img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
					img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1t.png",img1)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2t.png",img2)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3t.png",img3)
					img1 = np.clip(np.amax(real_model_float, axis=0)*256, 0,255).astype(np.uint8)
					img2 = np.clip(np.amax(real_model_float, axis=1)*256, 0,255).astype(np.uint8)
					img3 = np.clip(np.amax(real_model_float, axis=2)*256, 0,255).astype(np.uint8)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1i.png",img1)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2i.png",img2)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3i.png",img3)
					print("[sample]")
				
				if idx==batch_idxs-1:
					self.save(config.checkpoint_dir, epoch)
	
	def z2voxel(self, z):
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		dimc = self.cell_grid_size
		dimf = self.frame_grid_size
		
		frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
		queue = []
		
		frame_batch_num = int(dimf**3/self.batch_size)
		assert frame_batch_num>0
		
		#get frame grid values
		for i in range(frame_batch_num):
			model_out = self.sess.run(self.zG,
				feed_dict={
					self.z_vector: z,
					self.point_coord: self.frame_coords[i*self.batch_size:(i+1)*self.batch_size],
				})
			x_coords = self.frame_x[i*self.batch_size:(i+1)*self.batch_size]
			y_coords = self.frame_y[i*self.batch_size:(i+1)*self.batch_size]
			z_coords = self.frame_z[i*self.batch_size:(i+1)*self.batch_size]
			frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.batch_size])
		
		#get queue and fill up ones
		for i in range(1,dimf+1):
			for j in range(1,dimf+1):
				for k in range(1,dimf+1):
					maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					if maxv!=minv:
						queue.append((i,j,k))
					elif maxv==1:
						x_coords = self.cell_x+(i-1)*dimc
						y_coords = self.cell_y+(j-1)*dimc
						z_coords = self.cell_z+(k-1)*dimc
						model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
		
		print("running queue:",len(queue))
		cell_batch_size = dimc**3
		cell_batch_num = int(self.batch_size/cell_batch_size)
		assert cell_batch_num>0
		#run queue
		while len(queue)>0:
			batch_num = min(len(queue),cell_batch_num)
			point_list = []
			cell_coords = []
			for i in range(batch_num):
				point = queue.pop(0)
				point_list.append(point)
				cell_coords.append(self.cell_coords[point[0]-1,point[1]-1,point[2]-1])
			cell_coords = np.concatenate(cell_coords, axis=0)
			model_out_batch = self.sess.run(self.zG,
				feed_dict={
					self.z_vector: z,
					self.point_coord: cell_coords,
				})
			for i in range(batch_num):
				point = point_list[i]
				model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
				x_coords = self.cell_x+(point[0]-1)*dimc
				y_coords = self.cell_y+(point[1]-1)*dimc
				z_coords = self.cell_z+(point[2]-1)*dimc
				model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
				
				if np.max(model_out)>self.sampling_threshold:
					for i in range(-1,2):
						pi = point[0]+i
						if pi<=0 or pi>dimf: continue
						for j in range(-1,2):
							pj = point[1]+j
							if pj<=0 or pj>dimf: continue
							for k in range(-1,2):
								pk = point[2]+k
								if pk<=0 or pk>dimf: continue
								if (frame_flag[pi,pj,pk] == 0):
									frame_flag[pi,pj,pk] = 1
									queue.append((pi,pj,pk))
		return model_float
	
	#may introduce foldovers
	def optimize_mesh(self, vertices, z, iteration = 3):
		new_vertices = np.copy(vertices)
		new_v_out = self.sess.run(self.zG,
			feed_dict={
				self.z_vector: z,
				self.point_coord: new_vertices,
			})
		
		for iter in range(iteration):
			for i in [-1,0,1]:
				for j in [-1,0,1]:
					for k in [-1,0,1]:
						if i==0 and j==0 and k==0: continue
						offset = np.array([[i,j,k]],np.float32)/(self.real_size*6*2**iter)
						current_vertices = vertices+offset
						current_v_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: z,
								self.point_coord: current_vertices,
							})
						keep_flag = abs(current_v_out-self.sampling_threshold)<abs(new_v_out-self.sampling_threshold)
						new_vertices = current_vertices*keep_flag+new_vertices*(1-keep_flag)
						new_v_out = current_v_out*keep_flag+new_v_out*(1-keep_flag)
			vertices = new_vertices
		
		return vertices

	def test_interp(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		interp_size = 8
		idx1 = 0
		idx2 = 2
		
		batch_voxels1 = self.data_voxels[idx1:idx1+1]
		batch_voxels2 = self.data_voxels[idx2:idx2+1]
		
		model_z1 = self.sess.run(self.sE,
			feed_dict={
				self.vox3d: batch_voxels1,
			})
		model_z2 = self.sess.run(self.sE,
			feed_dict={
				self.vox3d: batch_voxels2,
			})
		
		for t in range(interp_size):
			tmp_z = model_z2*t/(interp_size-1) + model_z1*(interp_size-1-t)/(interp_size-1)
			model_float = self.z2voxel(tmp_z)
			#img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			#img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			#img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(config.sample_dir+"/"+"out"+str(t)+".ply", vertices, triangles)
			
			print("[sample interpolation]")
	
	def test(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		#test_num = self.data_voxel.shape[0]
		test_num = 16
		for t in range(test_num):
			print(t,test_num)

			batch_voxels = self.data_voxels[t:t+1]
			model_z = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			
			model_float = self.z2voxel(model_z)
			
			#img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			#img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			#img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(config.sample_dir+"/"+"out"+str(t)+".ply", vertices, triangles)
			
			print("[sample]")
	
	def get_z(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		hdf5_path = self.data_dir+'/'+self.dataset_name+'_z.hdf5'
		chair_num = len(self.data_voxels)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		hdf5_file.create_dataset("zs", [chair_num,self.z_dim], np.float32)

		for idx in range(0, chair_num):
			print(idx)
			batch_voxels = self.data_voxels[idx:idx+1]
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			hdf5_file["zs"][idx,:] = np.reshape(z_out,[self.z_dim])
			
		print(hdf5_file["zs"].shape)
		hdf5_file.close()
		print("[z]")
	
	def test_z(self, config, batch_z, dim):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		for t in range(batch_z.shape[0]):
			model_float = self.z2voxel(batch_z[t:t+1])
			#img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			#img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			#img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(config.sample_dir+"/"+"out"+str(t)+".ply", vertices, triangles)
			
			print("[sample GAN]")

	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.input_size)
			
	def save(self, checkpoint_dir, step):
		model_name = "IMAE.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
