from numpy import random

from network import network
from mnist import mnist as data

# Loading training and test sets
training_set, test_set = data.load()
print("Successfully loaded data. Initializing network...")

# Creating network
hidden_layers = [3]
x, y = training_set[0]
input_size = x.shape[0]
output_size = y.shape[0]

net = network(input_size, output_size, hidden_layers)
print("Successfully initialized network. Training network...")

# Training variables
learning_rate = 1
set_size = 150
num_cycles = 30

net.train(training_set, learning_rate, set_size, num_cycles)
print("Successfully trained network. Validating network...")

test_outputs=[]

for i in range(len(test_set)):
    x, y = test_set[i]
    test_outputs.append(net.forwardpass(x))
    
performance = data.validate(test_set, test_outputs) * 100
print(performance)