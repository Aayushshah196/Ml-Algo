import numpy as np
import math
import copy

from activation import ReLU, sigmoid, LeakyReLU
from optimizer import GradientDescent


class Layer:

	def layer_name(self):
		return self.__class__.__name__

	def set_optimizer(self, optimizer):
		self.optimizer = copy.copy(optimizer)

	def forward(self, A=None, weights=None, bias=None):
		raise NotImplementedError()

	def backward(self, gradients=None):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()


class Dense(Layer):

	def __init__(self, layer_size, input_shape=None, activation="relu"):
		self.layer_size = layer_size
		self.input_shape = input_shape
		self.optimizer = None
		self.activation = LeakyReLU() if activation=="relu" else sigmoid()
		self.input_activation = None
		self.trainable = True
		self.weights = None
		self.bias = None
		self.gradients = None
		self.z = None

	def set_input_shape(self, input_shape):
		self.input_shape = input_shape


	def initialize(self):
		"""
			The weights and biases won't be initialized until it is trained or specified
		"""
		#Xavier initialization in uniform distribution
		distribution_range = 6 / math.sqrt(self.input_shape[0]+self.layer_size)
		#weights_shape = (k, n), input_shape = (n, m) || --> m=batch size, n=size of input example, k = layer_size
		weights = np.random.uniform(0, distribution_range, (self.layer_size, self.input_shape[0]))*0.01
		# self.weights = np.random.randn(self.layer_size, self.input_shape[0])*0.01
		bias = np.zeros((self.layer_size, 1))

		return weights, bias

	def layer_name(self):
		return self.__class__.__name__

	def set_optimizer(self, optimizer):
		self.optimizer = copy.copy(optimizer)
		self.optimizer.getparams()

	def set_trainable(self, trainable):
		self.trainable = trainable

	def output_shape(self):
		return (self.layer_size, self.input_shape[0])

	def summary(self):
		trainable_params = self.layer_size * self.input_shape[0] + self.layer_size
		return self.layer_name(), self.output_shape(), trainable_params, self.weights.shape, self.bias.shape

	def getweights(self):
		print(self.layer_name)
		print(f"Weights : {self.weights}")
		print(f"Bias : {self.bias}")

	def forward(self, A, weights, bias):
		self.input_activation = A
		self.z = (weights@self.input_activation)+bias
		return self.activation(self.z)

	def backward(self, dA, weights):

		dz = self.activation.gradient(self.z) * dA
		dw = (dz@self.input_activation.T) / self.input_activation.shape[1]
		db = np.mean(dz, axis=1, keepdims=True)
		dA = weights.T@dz

		return dA, dw, db


	def getparams(self):
		print(f"{self.layer_name()} - ")
		self.optimizer.getparams()
		print()



def zero_padding(img, pad_h, pad_w):
	img = np.pad(img, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)))
	return img


class Conv2d(Layer):
	def __init__(self, input_shape=None, kernel_size=3, filters=8, padding="valid", stride=1, activation=ReLU()):
		"""
			input_shape = (m, n_h, n_w, n_c)
			kernel_size: int (size of filters)
			filters = int (no of channels)
			padding = "same" or "valid"
			stride = int
			activation = activation function
		"""

		self.input_shape = input_shape
		self.kernel_size = kernel_size
		self.filters = filters
		self.padding = padding
		self.stride = stride
		self.activation = ReLU() if activation=="relu" else sigmoid()
		self.input_activation = None
		self.pad_h = None
		self.pad_w = None
		self.z = None


	def initialize(self):
		"""
			The weights and biases won't be initialized until it is trained or specified
		"""
		#Xavier initialization in uniform distribution
		distribution_range = 6 / math.sqrt(2*self.filters)
		#weights_shape = (f, f, n_c_prev, n_c), input_shape = (m, n_h, n_w, n_c) || --> m=batch size, n=size of input example, k = layer_size
		weights = np.random.uniform(0, distribution_range, (self.kernel_size, self.kernel_size, self.input_shape[-1], self.filters))*0.1
		# self.weights = np.random.randn(self.layer_size, self.input_shape[0])*0.01
		bias = np.zeros((1, 1, self.input_shape[-1], self.filters))

		return weights, bias


	def set_input_shape(self, input_shape):
		self.input_shape = input_shape


	def forward(self, A_prev, weights, bias):
		self.input_activation = A_prev

		m, n_h_prev, n_w_prev, n_c_prev = self.input_shape

		pad_h = 0
		pad_w = 0
		if self.padding=="same" and self.pad_h is None:
			self.pad_h = int(((n_h_prev-1)*self.stride - n_h_prev + self.kernel_size)/2)
			self.pad_w = int(((n_w_prev-1)*self.stride - n_w_prev + self.kernel_size)/2)

		n_h = int((n_h_prev-self.kernel_size+2*self.pad_h)/self.stride) + 1
		n_w = int((n_w_prev-self.kernel_size+2*self.pad_w)/self.stride) + 1

		self.output_shape = (m, n_h, n_w, self.filters)
		output = np.zero(self.output_shape)

		A_prev = zero_padding(A_prev, pad_h, pad_w)


		for i in range(m):
			for h in range(n_h):
				for w in range(n_w):
					for c in range(self.filters):
						vert_start = h*self.stride
						vert_end = vert_start + self.kernel_size
						hori_start = w*self.stride
						hori_end = hori_start + self.kernel_size
						output[i,h,w,c] = self._conv_single(A_prev[i, vert_start:vert_end, hori_start:hori_end, :], weights[:,:,:,c], bias[:,:,:,c])


		self.z = self.activation(output)

		return self.z


	def backward(self, dA, weights):

		dz = self.activation.gradient(self.z)*dA
		m, n_h_prev, n_w_prev, n_c_prev = self.input_shape
		m, n_h, n_w, n_c = self.output_shape

		dA_prev = np.zero(self.input_shape)
		dW = np.zeros_like(weights)
		db = np.zeros((1, 1, n_c_prev, n_c))
		

		a_prev = zero_padding(self.input_activation, self.pad_h, self.pad_w)
		dA_prev_pad = zero_padding(self.dA_prev, self.pad_h, self.pad_w)

		for i in range(m):
			for h in range(n_h):
				for w in range(n_w):
					for c in range(self.filters):
						vert_start = h*self.stride
						vert_end = vert_start + self.kernel_size
						hori_start = w*self.stride
						hori_end = hori_start + self.kernel_size
						dA_prev_pad[i, vert_start:vert_end, hori_start:hori_end, :] += weights[:,:,:,c]*dz[i,h,w,c]
						dW += a_prev[i, vert_start:vert_end, hori_start:hori_end, :] * dz[m,h,w,c]
						db[:,:,:,c] += dz[i,h,w,c]

			dA_prev[i, :, :, :] = dA_prev_pad[i,self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]



		return dA_prev, dW, db			


	def _conv_single(self, img, weights, bias):
		z = np.multiply(img, weights) + bias
		return np.sum(z)


class MaxPool(Layer):

	def __init__(self, kernel_size, padding, stride):
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

	def forward(self, A_prev, W, b):
		self.input_activation = A_prev
		(m, n_h_prev, n_w_prev, n_c_prev) = self.input_shape

		pad_h = 0
		pad_w = 0
		if self.padding=="same" and self.pad_h is None:
			self.pad_h = int(((n_h_prev-1)*self.stride - n_h_prev + self.kernel_size)/2)
			self.pad_w = int(((n_w_prev-1)*self.stride - n_w_prev + self.kernel_size)/2)

		n_h = int((n_h_prev-self.kernel_size+2*self.pad_h)/self.stride) + 1
		n_w = int((n_w_prev-self.kernel_size+2*self.pad_w)/self.stride) + 1

		self.output_shape = (m, n_h,  n_w, n_c_prev)
		output = np.zeros(self.output_shape)
		self.A_prev = A_prev
		for i in range(m):
			for h in range(n_h):
				for w in range(n_w):
					for c in range(n_c_prev):
						vert_start = h*self.stride
						vert_end = vert_start + self.kernel_size
						hori_start = w*self.stride
						hori_end = hori_start + self.kernel_size
						output[i,h,w,c] = np.max(A_prev[i, vert_start:vert_end, hori_start:hori_end, c])

		return output


	def backward(self, dA, weights):
		dz = np.zeros(self.input_shape)
		(m, n_h_prev, n_w_prev, n_c_prev) = self.input_shape
		n_h = int((n_h_prev-self.kernel_size+2*self.pad_h)/self.stride) + 1
		n_w = int((n_w_prev-self.kernel_size+2*self.pad_w)/self.stride) + 1

		for i in range(m):
			for h in range(n_h):
				for w in range(n_w):
					for c in range(n_c_prev):
						vert_start = h*self.stride
						vert_end = vert_start + self.kernel_size
						hori_start = w*self.stride
						hori_end = hori_start + self.kernel_size
						index = np.argmax(self.A_prev[i, vert_start:vert_end, hori_start:hori_end, c])
						idx_0 = index/self.kernel_size
						idx_1 = index%self.kernel_size
						dz[i, vert_start+idx_0, hori_start+idx_1, c] = dA[i,h,w,c]

		return dz




class Flatten(Layer):
	def __init__(self, prev_shape=None):
		self.prev_shape = prev_shape

	def forward(self, A_prev, weights=None, bias=None):
		# Changes shape of A to (m, - )
		self.prev_shape = A.shape
		A = np.reshape(A, (A.shape[0], -1))
		# Transposes the shape of A to (- , m) for  Dense layer
		return A.T

	def backward(self):
		dA = dA.T
		return np.reshape(dA, (self.prev_shape.shape))
