from array import array
import numpy as np
import os
from struct import unpack

from data import data

class mnist(data):
	""" MNIST data helper class """

	def load_set (label_name, data_name):
		""" reads in and formats mnist data sets """
		MNIST_path = os.path.join('data_sets', 'MNIST')
		data_file = open(os.path.join(MNIST_path, data_name), 'rb')
		label_file = open(os.path.join(MNIST_path, label_name), 'rb')
		data_set = []

		magic, size = unpack(">II", label_file.read(8))
		magic, size, rows, columns = unpack(">IIII", data_file.read(16))

		images = array("B", data_file.read())
		labels = array("B", label_file.read())

		for i in range(size):
			y = np.zeros((10, 1))
			y[labels[i]] = 1

			x = np.matrix(images[i * rows * columns:
										(i + 1) * rows * columns]).transpose()

			data_set.append((x, y))

		return data_set

	@classmethod
	def load (cls):
		""" load MNIST training and test data sets """
		training_set = cls.load_set("train-labels.idx1-ubyte",
													"train-images.idx3-ubyte")
		test_set = cls.load_set("t10k-labels.idx1-ubyte",
													"t10k-images.idx3-ubyte")
		return training_set, test_set

	# Needs to be completed
	@classmethod
	def validate (cls, test_set, outputs):
		return