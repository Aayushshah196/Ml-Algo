import numpy as np

class Metric:
	def loss(self, y_true, y_pred):
		raise NotImplementedError()

	def gradient(self):
		raise NotImplementedError()

	def accuracy(self, y_true, y_pred):
		raise NotImplementedError()


class MSE(Metric):

	def loss(self, y_true, y_pred):
		if y_true.shape != y_pred.shape:
			print("unmatched shape of parameters")
			# raise ShapeError()
		return np.mean(0.5*(y_true - y_pred)**2, axis=1)

	def gradient(self, y_true, y_pred):
		grad = -1*(y_true-y_pred)
		return grad

	def accuracy(self, y_true, y_pred): 
		pass

