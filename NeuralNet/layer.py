import numpy as np
import math
import copy

from activation import ReLU, sigmoid, LeakyReLU
from optimizer import GradientDescent


class Layer(object):

	def layer_name(self):
		return self.__class__.__name__

	def set_optimizer(self, optimizer):
		self.optimizer = copy.copy(optimizer)

	def forward(self, X, trainable=True):
		raise NotImplementedError()

	def backward(self, gradients):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()


class Dense(Layer):

	def __init__(self, layer_size, input_shape=None, activation="relu"):
		self.layer_size = layer_size
		self.input_shape = input_shape
		self.optimizer = None
		self.activation = ReLU() if activation=="relu" else sigmoid()
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
		self.weights = np.random.uniform(0, distribution_range, (self.layer_size, self.input_shape[0]))*0.1
		# self.weights = np.random.randn(self.layer_size, self.input_shape[0])*0.01
		self.bias = np.zeros((self.layer_size, 1))

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


	def forward(self, A):
		self.input_activation = A
		self.z = (self.weights@self.input_activation)+self.bias
		return self.activation(self.z)


	def backward(self, dA):

		dz = self.activation.gradient(self.z) * dA
		dw = (dz@self.input_activation.T) / self.input_activation.shape[1]
		db = np.mean(dz, axis=1, keepdims=True)

		dA = self.weights.T@dz

		self.weights = self.optimizer.update(self.weights, dw)
		self.bias = self.optimizer.update(self.bias, db)

		return dA

	# def update(self, w, dw):


	def getparams(self):
		print(f"{self.layer_name()} - ")
		self.optimizer.getparams()
		print()

