import numpy as np

def batch_iterator(X, y, batch_size=32):
	n_samples = X.shape[1]
	X, y = shuffle_data(X, y)
	for i in np.arange(0, n_samples, batch_size):
		begin, end = i, min(i+batch_size, n_samples)

		yield X[:,begin:end], y[:,begin:end]

def shuffle_data(X, y, seed=None):
	if not seed:
		seed = X.shape[0]

	np.random.seed(seed)

	idx  = np.arange(X.shape[1])
	np.random.shuffle(idx)
	return X[:,idx], y[:,idx]