from numpy import random

from network import network
from mnist import mnist as data

# Loading training and test sets
training_set, test_set = data.load()
print("Successfully loaded data. Initializing network...")

# Creating network
hidden_layers = [30]
x, y = training_set[0]
input_size = x.shape[0]
output_size = y.shape[0]
net = network(input_size, output_size, hidden_layers)
print("Successfully initialized network. Training network...")

# Training variables
learning_rate = 1
set_size = 500
num_cycles = 1000

for i in range(num_cycles):
	random.shuffle(training_set)
	net.train(training_set[0:set_size], learning_rate)
print("Successfully trained network. Validating network...")


test_outputs = []
for x, y in test_set:
	test_outputs.append(net.forwardpass(x))

performance = data.validate(test_set, test_outputs)
print(performance)