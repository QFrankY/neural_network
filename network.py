import numpy as np

class network:
	""" Neural network class """

	def __init__ (self, input_size, output_size, hidden_layers):
		"""
		initialize network bias and weight matricies
		with sizes dependent on data set and hidden layer sizes
		"""
		self.layers = [input_size]
		self.layers.extend(hidden_layers)
		self.layers.append(output_size)
		self.biases = [np.random.randn(x, 1) for x in self.layers[1:]]
		self.weights = [np.random.randn(x, y) for x, y in 
								zip(self.layers[1:], self.layers[0:-1])]

	def sigmoid (self, z):
		return 1 / (1 + np.exp(-z))

	def sigmoid_prime (self, z):
		return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

	def forwardpass(self, x, softmax=False):
		""" passes numpy vector x through neural network layers """
		activation = x
		for i in range(len(self.layers) - 1):
			z = np.dot(self.weights[i], activation) + self.biases[i]
			activation = self.sigmoid(z)

		if softmax:
			activation = activation / np.sum(activation)

		return activation

	def backprop(self, x, y):
		""" determining delta weights and biases from error """
		delta_weights = [np.zeros(x.shape) for x in self.weights]
		delta_biases = [np.zeros(x.shape) for x in self.biases]

		z_values = []
		a_values = [x]

		""" forward pass """
		for i in range(len(self.layers) - 1):
			z = np.dot(self.weights[i], a_values[i]) + self.biases[i]
			z_values.append(z)
			a_values.append(self.sigmoid(z))

		""" error calculation """
		err = a_values[-1] - y # Based on cross entropy
		delta_biases[-1] = err
		delta_weights[-1] = np.dot(err, a_values[-2].transpose())

		""" back pass with error calculations """
		for i in range(1, len(z_values)):
			err = np.multiply(np.dot(self.weights[-i].transpose(), err),
					self.sigmoid_prime(z_values[-i-1]))

			delta_biases[-i-1] = err
			delta_weights[-i-1] = np.dot(err, a_values[-i-2].transpose())

		return delta_biases, delta_weights

	def train(self, training_data, learning_rate, set_size, num_cycles):
		""" cycle through training data then update network """
		num_sets = int(len(training_data) / set_size)

		for i in range(num_cycles):
			for j in range(num_sets):
				data_set = training_data[j * set_size: (j + 1) * set_size]
				d_w_total = [np.zeros(x.shape) for x in self.weights]
				d_b_total = [np.zeros(x.shape) for x in self.biases]

				for x, y in data_set:
					d_b, d_w = self.backprop(x, y)
					d_w_total = [old + new for old, new in zip(d_w_total, d_w)]
					d_b_total = [old + new for old, new in zip(d_b_total, d_b)]

				self.weights = [old - np.multiply(learning_rate / len(data_set), 
						total) for old, total in zip(self.weights, d_w_total)]
				self.biases = [old - np.multiply(learning_rate / len(data_set), 
						total) for old, total in zip(self.biases, d_b_total)]
			np.random.shuffle(training_data)