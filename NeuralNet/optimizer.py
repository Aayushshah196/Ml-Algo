import numpy as np

class GradientDescent:

	def __init__(self, learning_rate=0.000009):
		self.learning_rate = learning_rate
		self.count = 0

	def getparams(self):
		print(self.learning_rate)
		print(self.count)

	def update(self, W, dW):
		self.count += 1
		# print("Updating parameters")
		if W.shape != dW.shape:
			print("unmatched shape of parameters")
			pass
		if(self.count%10000==0):
			self.learning_rate = self.learning_rate/5.0
		W = W - self.learning_rate*dW
		return W

# class GradientDescent:
# 	def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
# 		self.learning_rate = learning_rate
# 		self.eps = 1e-8
# 		self.m = None
# 		self.v = None
# 		# Decay rates
# 		self.b1 = b1
# 		self.count = 0
# 		self.b2 = b2

# 	def set_learningrate(self, rate):
# 		self.learning_rate = rate 

# 	def update(self, w, dw):
# 		self.count += 1
# 	# If not initialized
# 		if self.m is None:
# 			self.m = np.zeros(np.shape(dw))
# 			self.v = np.zeros(np.shape(dw))

# 		self.m = self.b1 * self.m + (1 - self.b1) * dw
# 		self.v = self.b2 * self.v + (1 - self.b2) * np.power(dw, 2)

# 		m_hat = self.m / (1 - self.b1)
# 		v_hat = self.v / (1 - self.b2)

# 		self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

# 		return w - self.w_updt

# 	def getparams(self):
# 		print(self.learning_rate)
# 		print(self.count)

# class Adam:
# 	self.