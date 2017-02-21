import numpy as np

from data import data

class encoder(data):
	""" 5 input/output auto encoder """

	@classmethod
	def load (cls):
		training_set = []
		test_set = []

		for i in range(5):
			vector = np.zeros((5, 1))
			vector[i] = 1

			training_set.append((vector, vector))
			test_set.append((vector, i))

		return training_set, test_set

	@classmethod
	def validate(cls, test_set, outputs):
		num_valid = 0

		for i in range(len(test_set)):
			x, y = test_set[i]
			if y == np.argmax(outputs[i]):
				num_valid = num_valid + 1

		return num_valid / len(test_set)