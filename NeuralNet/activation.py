import numpy as np

class ReLU:
	def __call__(self, z):
		return np.maximum(0,z)

	def gradient(self, z):
		return np.where(z>0, 1, 0)

class sigmoid:
	"""
		sigmoid(x) = 1 / ( 1 - exp(-x))
	"""
	def __call__(self, z):
		return 1.0 / (1+ np.exp(-z))

	def gradient(self, z):
		"""
		d(sigmoid(x)) = 1/(1 + exp(-x)) - 1/(1 + exp(-x))^2 
		"""
		p = self.__call__(z)
		return p*(1-p)

class LeakyReLU:
	def __init__(self, alpha=0.01):
		self.alpha = alpha
		
	def __call__(self, z):
		return np.maximum(self.alpha*z, z)

	def gradient(self, z):
		return np.where(z>0, 1, self.alpha)

class softmax:
	def __call__(self, z):
		return np.exp(z)/np.sum(np.exp(z))

	def gradient(self, z):
		p = self.__call__(z)
		return p*(1-p)