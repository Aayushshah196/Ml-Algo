import numpy as np
import pandas as pd


class KNN:
	def __init__(self, n = 2):
		self.n = n
		self.history = {}

	def fit(self, data, n):
		self.n = n
		self.data = data

	def train(self, epochs = 1000):
		self.centroids = _initialise_centroids()
		self.history["centroids"] = []

		for i in range(epochs):

			knn_sum = np.shape(shape=(self.n, self.data.shape[1]))
			knn_count = np.zeros(shape=(self.n, 1))

			for x in self.data:
				dist = np.argmin(((self.centroids-x)**2).sum(axis=1))
				knn_sum[dist] = knn_sum[dist] + x
				knn_count[dist] += 1

			self.centroids = knn_sum/ (knn_count + 0.00001)
			if i%100 ==0 :
				history["centroids"].append(self.centroids)

		def _initialise_centroids(self):
			centroids = np.empty(shape=(self.n, self.data.shape[1]))
			for i in range(self.data.shape[1]):
				low = self.data[:,i].min()
				high = self.data[:,i].max()
				centroids[:,i] =  np.random.uniform(low=low, high=high, size=centroids[:,i].shape)

			return centroids
