from layer import Dense
from optimizer import GradientDescent
from activation import ReLU, sigmoid
from metrics import MSE
from utils import batch_iterator

from prettyTable import PrettyTable

class Module:

    def fit(self):
        raise NotImplementedError()

    def train(self, X, y, batch_size, epochs):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class NeuralNet:

    def __init__(self, optimizer, metrics):
        self.optimizer = optimizer
        self.metrics = metrics()
        self.layers = []
        self._parameters = {}

    def add(self, layer):
        layer.set_optimizer(self.optimizer)
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())
        self.layers.append(layer)


    def fit(self):
        for layer in self.layers:
            layer.initialize()

    def train(self, X, y, batch_size=32, epochs=100):
        loss = None
        y_pred = None

        for i in range(epochs):
            for batch_x, batch_y in batch_iterator(X, y, batch_size):
                y_pred = self._forward(batch_x)
                loss = self.metrics.loss(batch_y, y_pred)
                gradient = self.metrics.gradient(batch_y, y_pred)
                self._backward(gradient)

            if i%10 == 0:
                print()
                print(f" -> Loss : {loss} {'----------'*3} at {i} / {epochs} ")


    def fit_and_train(self, X, y, batch_size=32, epochs=500):
        self.fit()
        self.train(X, y, batch_size, epochs)



    def _forward(self, X):
        layer_output = X

        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        return layer_output


    def _backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


    def predict(self, X):
        return self.forward(X)
    
    def getweights(self):
        for layer in self.layers:
            layer.getweights()

    def summary(self):
        table = PrettyTable()
        
        table.field_names = ["Layer No.", "Layer Name", "Layer Size", "Trainable Parameters", "Weights", "Bias"]

        for i, layer in enumerate(self.layers):
            layer_name, layer_size, trainable_params, weights, bias = layer.summary()
            table.add_row([i+1, layer_name, layer_size, trainable_params, weights, bias])

        print(table)

# shape of X = (n, m) => n=no of axes, m=no. of examples
# layer_size, input_shape=None, activation=ReLU, optimizer=GradientDescent)
# optimizer, metrics
if __name__ == "__main__":
	optimizer = GradientDescent()
	model = NeuralNet(optimizer, metrics=MSE)

	model.add(Dense(8, (4,None)))
	model.add(Dense(4))
	model.add(Dense(1))