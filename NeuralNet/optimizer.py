import numpy as np


class GradientDescent:
	def __init__(self, learning_rate=0.000009):
		self.learning_rate = learning_rate
		self.count = 0

	def getparams(self):
		print(self.learning_rate)
		print(self.count)

	def update_weights(self, W, dW):
		self.count += 1
		# print("Updating parameters")
		if W.shape != dW.shape:
			print("unmatched shape of parameters")

		return W - self.learning_rate*dW

	def update_bias(self, b, db):
		self.count += 1
		# print("Updating parameters")
		if b.shape != db.shape:
			print("unmatched shape of parameters")
			
		return b - self.learning_rate*db



class Momentum:
	"""
		vdw - the exponentially weighted average of past gradients
		beta - hyperparameter to be tuned
		dW - cost gradient with respect to current layer
		W - the weight matrix (parameter to be updated)
		ϵ(0.000005) - very small value to avoid dividing by zero
	"""
	def __init__(self, beta, learning_rate=0.0000009):
		self.learning_rate = learning_rate
		self.beta = beta
		self.vdw = None
		self.vdb = None


	def update_weights(self, W, dW):
		if (self.vdw==None):
			self.vdw = np.zeros_like(W)

		self.vdw = self.beat*self.vdw + (1-self.beta)*dW

	def update_bias(self, b, db):
		if (self.vdb==None):
			self.vdb = np.zeros_like(b)

		self.vdb = self.beat*self.vdb + (1-self.beta)*db



class RMSProp:
	"""
		sdw - the exponentially weighted average of past squares of gradients
		beta - hyperparameter to be tuned
		dW - cost gradient with respect to current layer
		W - the weight matrix (parameter to be updated)
		ϵ(0.000005) - very small value to avoid dividing by zero
	"""
	def __init__(self, beta, learning_rate=0.0000009):
		self.learning_rate = learning_rate
		self.beta = beta
		# exponentially weighted average of past squares of gradients
		self.sdw = None
		self.sdb = None

	def update_weights(self, W,  dW):
		if self.sdw==None:
			self.sdw = np.zeros_like(W)

		self.sdw = self.beta*self.sdw + (1-self.beta)*np.power(dW, 2)
		return W - self.learning_rate*dW/np.sqrt(self.sdw+0.000005)

	def update_bias(self, b,  db):
		if self.sdb==None:
			self.sdb = np.zeros_like(b)

		self.sdb = self.beta*self.sdb + (1-self.beta)*np.power(db, 2)
		return b - self.learning_rate*db/np.sqrt(self.sdb+0.000005)



class Adam:
	"""
		vdw - the exponentially weighted average of past gradients
		sdw - the exponentially weighted average of past squares of gradients
		beta1 - hyperparameter to be tuned
		beta2 - hyperparameter to be tuned
		dW - cost gradient with respect to current layer
		W - the weight matrix (parameter to be updated)
		ϵ(0.000005) - very small value to avoid dividing by zero
	"""
	def __init__(self, learning_rate, beta1, beta2):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.vdw = None
		self.sdw = None
		self.vdb = None
		self.sdb = None

	def update_weights(self, W, dW):
		if self.vdw is None:
			self.vdw = np.zeros_like(dW)
			self.sdw = np.zeros_like(dW)

		self.vdw = self.beta1*self.vdw + (1-self.beta1)*dW
		self.sdw = self.beta2*self.sdw + (1-self.beta2)*np.power(dW, 2)

		vdw_corr = self.vdw / np.power((1-self.beta1), 1)
		sdw_corr = self.sdw / np.power((1-self.beta2), 1)

		return W - self.learning_rate*vdw_corr/np.sqrt(sdw_corr+0.000005)

	def update_bias(self, b, db):
		if self.vdb is None:
			self.vdb = np.zeros_like(db)
			self.sdb = np.zeros_like(db)

		self.vdb = self.beta1*self.vdb + (1-self.beta1)*db
		self.sdb = self.beta2*self.sdb + (1-self.beta2)*np.power(db, 2)

		vdb_corr = self.vdb / np.power((1-self.beta1), 1)
		sdb_corr = self.sdb / np.power((1-self.beta2), 1)

		return b - self.learning_rate*vdb_corr/np.sqrt(sdb_corr+0.000005)