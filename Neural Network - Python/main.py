# NEURAL NETWORK TRAINING

# Importing required libraries and functions
import numpy as np
import neural as nn
import sklearn.datasets
import matplotlib.pyplot as plt


# Creating a data set using sklearn
X, Y = sklearn.datasets.make_moons(n_samples = 500, noise = 0.15)
X, Y = X.T, Y.reshape(1, Y.shape[0])

# Neural network training process
parameters, cost = nn.train(X, Y, 0.7, 12, 2000)

# Drawing cost chart
plt.figure(1, figsize=(10, 6))
plt.title('COST CHART')
plt.plot(cost)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()