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

		return size, images, labels, rows * columns

	@classmethod
	def load (cls):
		""" load MNIST training and test data sets """
		training_set = []
		test_set = []

		size, images, labels, pixels = cls.load_set("train-labels.idx1-ubyte",
													"train-images.idx3-ubyte")
		for i in range(size):
			y = np.zeros((10, 1))
			y[labels[i]] = 1
			x = np.matrix(images[i * pixels: (i + 1) * pixels]).transpose()
			training_set.append((x, y))

		size, images, labels, pixels = cls.load_set("train-labels.idx1-ubyte",
													"train-images.idx3-ubyte")
		for i in range(size):
			x = np.matrix(images[i * pixels: (i + 1) * pixels]).transpose()
			test_set.append((x, labels[i]))

		return training_set, test_set

	# Needs to be completed
	@classmethod
	def validate (cls, test_set, outputs):
		num_valid = 0

		for i in range(len(test_set)):
			x, y = test_set[i]
			if np.argmax(y) == np.argmax(outputs[i]):
				num_valid = num_valid + 1

		return num_valid / len(test_set)