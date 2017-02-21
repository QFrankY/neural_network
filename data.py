from abc import ABC, abstractmethod

class data(ABC):
	""" abstract data set class """

	@abstractmethod
	def load (cls):
		""" this is a class method and must be overide with @classmethod """
		pass

	@abstractmethod
	def validate (cls, test_set, outputs):
		""" this is a class method and must be overide with @classmethod """
		pass