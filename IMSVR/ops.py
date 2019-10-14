import numpy as np 
import tensorflow as tf

def lrelu(x, leak=0.02):
	return tf.maximum(x, leak*x)

def batch_norm(input, phase_train):
	return tf.contrib.layers.batch_norm(input, decay=0.99, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_train)

def instance_norm(input, phase_train):
	return tf.contrib.layers.instance_norm(input)

def linear(input_, output_size, scope):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
		bias = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer())
		print("linear","in",shape,"out",(shape[0],output_size))
		return tf.matmul(input_, matrix) + bias

def conv2d(input_, shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable('Matrix', shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
		bias = tf.get_variable('bias', [shape[-1]], initializer=tf.zeros_initializer())
		conv = tf.nn.conv2d(input_, matrix, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv2d","in",input_.shape,"out",conv.shape)
		return conv

def conv3d(input_, shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", shape, initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [shape[-1]], initializer=tf.zeros_initializer())
		conv = tf.nn.conv3d(input_, matrix, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv3d","in",input_.shape,"out",conv.shape)
		return conv

def conv2d_nobias(input_, shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable('Matrix', shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
		conv = tf.nn.conv2d(input_, matrix, strides=strides, padding=padding)
		print("conv2d","in",input_.shape,"out",conv.shape)
		return conv

def write_ply(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()

def write_ply_point_normal(name, vertices, normals=None):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("end_header\n")
	if normals is None:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()