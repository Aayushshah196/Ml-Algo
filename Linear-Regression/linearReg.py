]import numpy as np

class LinearRegression():
	def __init__(self):
		self.batch_size = 5
		self.loss = []

	def train(self, x, y, paramCount = 1, epochs = 10000, learning_rate = 0.0001):
		self.learning_rate = learning_rate
		self._coeffCount = paramCount
		self.weights = np.random.rand(1)
		self.bias = np.random.rand(1)

		for epoch in range(epochs):
			y1 = self._forward(x)
			# print(y1)
			dw, db = self._backward(x, y, y1)
			self._update_weights(dw, db)
			if(epoch%200 == 0):
				self.learning_rate /= 1.1
		print(y)	
		print(self._forward(x))


	def predict(self, x):
		y1 = self.weights * x + self.bias
		return y1


	def _forward(self, x):
		y1 = self.weights * x + self.bias
		return y1


	def _backward(self, x, y, y1):
		self.loss.append(np.mean(y - y1))
		dw = -2 * np.mean((y-y1)*x)
		# print(dw)
		db = -2 * np.mean(y - y1)
		# print(dw.shape)
		# print(db.shape)
		return dw, db


	def _update_weights(self, dw, db):
		self.weights = self.weights - self.learning_rate * dw
		self.bias = self.bias - self.learning_rate * db

	def getWeights(self):
		print(self.weights)
		print(self.bias)



if __name__ == "__main__":
	x = np.array([1,2,3,4,5,6,7,8,9,10])
	y = 3*x + 1
	model = LinearRegression()
	model.train(x, y)
	model.getWeights()



# class LinearRegression():
# 	def __init__(self):
# 		self.batch_size = 5
# 		self.loss = []

# 	def train(self, x, y, paramCount = 1, epochs = 10, learning_rate = 0.0001):
# 		self.learning_rate = learning_rate
# 		self._coeffCount = paramCount
# 		self.weights = np.array([[1.25]])#np.random.rand(1, self._coeffCount)
# 		self.bias = np.array([[5.2]])#np.random.rand(1,1)

# 		for _ in range(epochs):
# 			y1 = self._forward(x)
# 			# print(y1)
# 			dw, db = self._backward(x, y, y1)
# 			self._update_weights(dw, db)
# 		print(y)	
# 		print(self._forward(x))


# 	def predict(self, x):
# 		y1 = np.matmul(x, np.transpose(self.weights)) + self.bias
# 		return y1


# 	def _forward(self, x):
# 		y1 = np.matmul(x, np.transpose(self.weights)) + self.bias
# 		return y1


# 	def _backward(self, x, y, y1):
# 		# print(x.shape)
# 		# print(y.shape)
# 		# print(y1.shape)
# 		self.loss.append(np.mean(y - y1))
# 		dw = 2* np.dot(np.transpose(y-y1), x)
# 		# print(dw)
# 		db = 2 * np.dot(np.transpose(y-y1), np.matmul(x, np.transpose(self.weights))-y)
# 		# print(dw.shape)
# 		# print(db.shape)
# 		return dw, db


# 	def _update_weights(self, dw, db):
# 		self.weights = self.weights - self.learning_rate * dw
# 		self.bias = self.bias - self.learning_rate * db

# 	def getWeights(self):
# 		print(self.weights)
# 		print(self.bias)
