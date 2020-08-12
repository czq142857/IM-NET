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

class IMSVR(object):
	def __init__(self, sess, is_training = False, z_dim=256, ef_dim=64, gf_dim=128, dataset_name='default', checkpoint_dir=None, pretrained_z_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		
		self.real_size = 64
		self.batch_size_input = 16*16*16*4
		self.batch_size = 16*16*16*4
		self.z_batch_size = 64
		
		self.view_size = 137
		self.crop_size = 128
		self.view_num = 24
		self.crop_edge = self.view_size-self.crop_size

		self.z_dim = z_dim

		self.ef_dim = ef_dim
		self.gf_dim = gf_dim

		self.dataset_name = dataset_name
		self.dataset_load = dataset_name + '_train'
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		if not is_training:
			self.dataset_load = dataset_name + '_test'

		if os.path.exists(self.data_dir+'/'+self.dataset_load+'.hdf5'):
			self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_load+'.hdf5', 'r')
			self.data_points_int = self.data_dict['points_'+str(self.real_size)][:]
			self.data_points = (self.data_points_int.astype(np.float32)+0.5)/256-0.5
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_pixel = self.data_dict['pixels'][:]
			data_dict_z = h5py.File(pretrained_z_dir, 'r')
			self.data_z = data_dict_z['zs'][:]
			if self.batch_size_input!=self.data_points.shape[1]:
				print("error: batch_size!=data_points.shape")
				exit(0)
			if self.view_num!=self.data_pixel.shape[1] or self.view_size!=self.data_pixel.shape[2]:
				print("error: view_size!=self.data_pixel.shape")
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
		#for test
		self.point_coord = tf.placeholder(shape=[None,3], dtype=tf.float32)
		self.z_vector_test = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32)
		self.view_test = tf.placeholder(shape=[1,self.crop_size,self.crop_size,1], dtype=tf.float32)
		
		#for train
		self.view = tf.placeholder(shape=[self.z_batch_size,self.crop_size,self.crop_size,1], dtype=tf.float32)
		self.z_vector = tf.placeholder(shape=[self.z_batch_size,self.z_dim], dtype=tf.float32)
		self.E = self.encoder(self.view, phase_train=True, reuse=False)
		self.loss = tf.reduce_mean(tf.square(self.z_vector - self.E))
		
		#for test
		self.sE = self.encoder(self.view_test, phase_train=False, reuse=True)
		self.zG = self.generator(self.point_coord, self.z_vector_test, phase_train=False, reuse=False)
		
		self.vars = tf.trainable_variables()
		self.g_vars = [var for var in self.vars if 'simple_net' in var.name]
		self.e_vars = [var for var in self.vars if 'encoder' in var.name]
		
	
	
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
			#h7 = tf.maximum(tf.minimum(h7, h7*0.01+0.99), h7*0.01)
			h7 = tf.maximum(tf.minimum(h7, 1), 0)
			
			return tf.reshape(h7, [batch_size,1])
	
	def encoder(self, view, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			#mimic resnet
			def resnet_block(input, dim_in, dim_out, scope):
				if dim_in == dim_out:
					output = conv2d_nobias(input, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_1')
					output = batch_norm(output, phase_train)
					output = lrelu(output)
					output = conv2d_nobias(output, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_2')
					output = batch_norm(output, phase_train)
					output = output + input
					output = lrelu(output)
				else:
					output = conv2d_nobias(input, shape=[3, 3, dim_in, dim_out], strides=[1,2,2,1], scope=scope+'_1')
					output = batch_norm(output, phase_train)
					output = lrelu(output)
					output = conv2d_nobias(output, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_2')
					output = batch_norm(output, phase_train)
					input_ = conv2d_nobias(input, shape=[1, 1, dim_in, dim_out], strides=[1,2,2,1], scope=scope+'_3')
					input_ = batch_norm(input_, phase_train)
					output = output + input_
					output = lrelu(output)
				return output
			
			view = 1.0 - view
			layer_0 = conv2d_nobias(view, shape=[7, 7, 1, self.ef_dim], strides=[1,2,2,1], scope='conv0')
			layer_0 = batch_norm(layer_0, phase_train)
			layer_0 = lrelu(layer_0)
			#no maxpool
			
			layer_1 = resnet_block(layer_0, self.ef_dim, self.ef_dim, 'conv1')
			layer_2 = resnet_block(layer_1, self.ef_dim, self.ef_dim, 'conv2')
			
			layer_3 = resnet_block(layer_2, self.ef_dim, self.ef_dim*2, 'conv3')
			layer_4 = resnet_block(layer_3, self.ef_dim*2, self.ef_dim*2, 'conv4')
			
			layer_5 = resnet_block(layer_4, self.ef_dim*2, self.ef_dim*4, 'conv5')
			layer_6 = resnet_block(layer_5, self.ef_dim*4, self.ef_dim*4, 'conv6')
			
			layer_7 = resnet_block(layer_6, self.ef_dim*4, self.ef_dim*8, 'conv7')
			layer_8 = resnet_block(layer_7, self.ef_dim*8, self.ef_dim*8, 'conv8')
			
			layer_9 = conv2d_nobias(layer_8, shape=[4, 4, self.ef_dim*8, self.ef_dim*8], strides=[1,2,2,1], scope='conv9')
			layer_9 = batch_norm(layer_9, phase_train)
			layer_9 = lrelu(layer_9)
			
			layer_10 = conv2d(layer_9, shape=[4, 4, self.ef_dim*8, self.z_dim], strides=[1,1,1,1], scope='conv10', padding="VALID")
			layer_10 = tf.nn.sigmoid(layer_10)
			
			return tf.reshape(layer_10, [-1,self.z_dim])
	
	def train(self, config):
		#first time run
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, ).minimize(self.loss, var_list=self.e_vars)
		self.sess.run(tf.global_variables_initializer())
		
		self.saver = tf.train.Saver(self.g_vars)
		could_load, checkpoint_counter = self.load_pretrained(config.pretrained_model_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			exit(0)
		
		batch_index_list = np.arange(len(self.data_points))
		batch_idxs = int(len(self.data_points)/self.z_batch_size)
		batch_num = int(self.batch_size_input/self.batch_size)
		if self.batch_size_input%self.batch_size != 0:
			print("batch_size_input % batch_size != 0")
			exit(0)
		self.saver = tf.train.Saver(max_to_keep=10)
		counter = 0
		start_time = time.time()
		
		
		batch_view = np.zeros([self.z_batch_size,self.crop_size,self.crop_size,1], np.float32)
		for epoch in range(counter, config.epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(0, batch_idxs):
				for t in range(self.z_batch_size):
					dxb = batch_index_list[idx*self.z_batch_size+t]
					which_view = np.random.randint(self.view_num)
					batch_view_ = self.data_pixel[dxb,which_view]
					#offset_x = np.random.randint(self.crop_edge)
					#offset_y = np.random.randint(self.crop_edge)
					offset_x = int(self.crop_edge/2)
					offset_y = int(self.crop_edge/2)
					if np.random.randint(2)==0:
						batch_view_ = batch_view_[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
					else:
						batch_view_ = np.flip(batch_view_[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size], 1)
					batch_view[t] = np.reshape(batch_view_/255.0, [self.crop_size,self.crop_size,1])
				batch_z = self.data_z[batch_index_list[idx*self.z_batch_size:(idx+1)*self.z_batch_size]]
				
				# Update AE network
				_, errAE = self.sess.run([ae_optim, self.loss],
					feed_dict={
						self.view: batch_view,
						self.z_vector: batch_z,
					})
				
				avg_loss += errAE
				avg_num += 1
				
				if idx==batch_idxs-1:
					print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, avgloss: %.8f" % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errAE, avg_loss/avg_num))
					
					if epoch%10 == 0:
						
						sample_z = self.sess.run(self.sE,
							feed_dict={
								self.view_test: batch_view[0:1],
							})

						model_float = np.zeros([256,256,256],np.float32)
						real_model_float = np.zeros([256,256,256],np.float32)
						dxb = batch_index_list[idx*self.z_batch_size]
						for minib in range(batch_num):
							batch_points_int = self.data_points_int[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
							batch_points = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
							batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
							
							model_out = self.sess.run(self.zG,
								feed_dict={
									self.z_vector_test: sample_z,
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
						img1 = (np.reshape(batch_view[0:1], [self.crop_size,self.crop_size])*255).astype(np.uint8)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_v.png",img1)
						print("[sample]")
						
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
					self.z_vector_test: z,
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
					self.z_vector_test: z,
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
				self.z_vector_test: z,
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
								self.z_vector_test: z,
								self.point_coord: current_vertices,
							})
						keep_flag = abs(current_v_out-self.sampling_threshold)<abs(new_v_out-self.sampling_threshold)
						new_vertices = current_vertices*keep_flag+new_vertices*(1-keep_flag)
						new_v_out = current_v_out*keep_flag+new_v_out*(1-keep_flag)
			vertices = new_vertices
		
		return vertices
		
	
	def test_interp(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		interp_size = 8
		
		idx1 = 0
		idx2 = 1
		
		add_out = "./out/"
		if not os.path.exists(add_out): os.makedirs(add_out)
		
		offset_x = int(self.crop_edge/2)
		offset_y = int(self.crop_edge/2)
		batch_view1 = self.data_pixel[idx1,0]
		batch_view1 = batch_view1[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
		batch_view1 = np.reshape(batch_view1/255.0, [1,self.crop_size,self.crop_size,1])
		batch_view2 = self.data_pixel[idx2,0]
		batch_view2 = batch_view2[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
		batch_view2 = np.reshape(batch_view2/255.0, [1,self.crop_size,self.crop_size,1])
		
		model_z1 = self.sess.run(self.sE,
			feed_dict={
				self.view_test: batch_view1,
			})
		model_z2 = self.sess.run(self.sE,
			feed_dict={
				self.view_test: batch_view2,
			})
		
		for t in range(interp_size):
			tmp_z = model_z2*t/(interp_size-1) + model_z1*(interp_size-1-t)/(interp_size-1)
			model_float = self.z2voxel(tmp_z)
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(add_out+str(t)+".ply", vertices, triangles)
			print("[sample interpolation]")

	def test(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		add_out = "./out/"
		add_image = "./image/"
		if not os.path.exists(add_out): os.makedirs(add_out)
		if not os.path.exists(add_image): os.makedirs(add_image)
		
		offset_x = int(self.crop_edge/2)
		offset_y = int(self.crop_edge/2)
		
		#test_num = self.data_pixel.shape[0]
		test_num = 16
		for t in range(test_num):
			print(t,test_num)
			
			batch_view = self.data_pixel[t,0]
			batch_view = batch_view[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
			batch_view = np.reshape(batch_view/255.0, [1,self.crop_size,self.crop_size,1])
			
			model_z = self.sess.run(self.sE,
				feed_dict={
					self.view_test: batch_view,
				})
			
			model_float = self.z2voxel(model_z)
			'''
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			img1 = (np.reshape(batch_view, [self.crop_size,self.crop_size])*255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_v.png",img1)
			'''
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(add_out+str(t)+".ply", vertices, triangles)
			
			cv2.imwrite(add_image+str(t)+".png", self.data_pixel[t,0])
			
			print("[sample image]")

	def test_image(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		add_out = "./out/"
		add_image = "./image/"
		if not os.path.exists(add_out): os.makedirs(add_out)
		if not os.path.exists(add_image):
			print("ERROR: image folder does not exist: ", add_image)
			return

		test_num = 16
		for t in range(test_num):
			img_add = add_image+str(t)+".png"
			print(t,test_num,img_add)
			
			if not os.path.exists(img_add):
				print("ERROR: image does not exist: ", img_add)
				return
			imgo_ = cv2.imread(img_add, cv2.IMREAD_GRAYSCALE)
			batch_view = cv2.resize(imgo_, (self.crop_size,self.crop_size))
			batch_view = np.reshape(batch_view/255.0, [1,self.crop_size,self.crop_size,1])
			
			model_z = self.sess.run(self.sE,
				feed_dict={
					self.view_test: batch_view,
				})
			
			model_float = self.z2voxel(model_z)

			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(add_out+str(t)+".ply", vertices, triangles)
			
			print("[sample image]")

	def test_all(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		num_of_points = 4096
		add_out = "./test_out/"
		if not os.path.exists(add_out): os.makedirs(add_out)
		
		offset_x = int(self.crop_edge/2)
		offset_y = int(self.crop_edge/2)
		
		test_num = self.data_pixel.shape[0]
		for t in range(test_num):
			print(t,test_num)
			
			batch_view = self.data_pixel[t,23]
			batch_view = batch_view[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
			batch_view = np.reshape(batch_view/255.0, [1,self.crop_size,self.crop_size,1])
			
			model_z = self.sess.run(self.sE,
				feed_dict={
					self.view_test: batch_view,
				})
			
			model_float = self.z2voxel(model_z)
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)

			#save mesh
			write_ply(add_out+str(t)+".ply", vertices, triangles)


	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.crop_size)
			
	def save(self, checkpoint_dir, step):
		model_name = "IMSVR.model"
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

	def load_pretrained(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")

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

