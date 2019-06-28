# FILE CONTAINING FUNCTIONS RELATED TO NEURAL NETWORK

# Importing important functions
import activation as act
import numpy as np


# FUNCTION THAT DEFINES NEURAL NETWORK'S STRUCTURE
def nnDefine(X, Y, hidden_size):
    np.random.seed(3)
    input_size = X.shape[0]
    output_size = Y.shape[0]
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((output_size, 1))
    return {'W1' : W1, 'W2' : W2, 'b1' : b1, 'b2' : b2}


# FUNCTION THAT CALCULATES NET'S FORWARD PROPAGATION
def propagate(X, param):
    X1 = np.dot(param['W1'], X) + param['b1']
    Y1 = act.tanh(X1)
    X2 = np.dot(param['W2'], Y1) + param['b2']
    y = act.sigmoid(X2)
    return y, {'X1' : X1, 'X2' : X2, 'Y1' : Y1, 'y' : y}


# COST FUNCTION
def cost_function(predicted, target):
    m = target.shape[1]
    eval = -np.sum(np.multiply(np.log(predicted), target) + np.multiply((1 - target), np.log(1 - predicted)))/m
    return np.squeeze(eval)


# BACK PROPAGATION
def back_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dW2 = (1 / m) * np.dot(dy, np.transpose(cache['Y1']))
    db2 = (1 / m) * np.sum(dy, axis = 1, keepdims = True)
    dX1 = np.dot(np.transpose(parameters['W2']), dy) * (1 - np.power(cache['Y1'], 2))
    dW1 = (1 / m) * np.dot(dX1, np.transpose(X))
    db1 = (1 / m) * np.sum(dX1, axis = 1, keepdims = True)
    return {'dW1' : dW1, 'db1' : db1, 'dW2' : dW2, 'db2' : db2}


# FUNCTION THAT UPDATES WEIGHTS AND BIASES
def update(gradient, parameters, learning_rate = 1.1):
    W1 = parameters['W1'] - learning_rate * gradient['dW1']
    b1 = parameters['b1'] - learning_rate * gradient['db1']
    W2 = parameters['W2'] - learning_rate * gradient['dW2']
    b2 = parameters['b2'] - learning_rate * gradient['db2']
    return {'W1' : W1, 'W2' : W2, 'b1' : b1, 'b2' : b2}


# FUNCTION RESPONSIBLE FOR TRAINING THE NEURAL NETWORK
def train(X, Y, learning_rate, hidden_size, iterations = 4000):
    parameters = nnDefine(X, Y, hidden_size)
    nn_cost = []
    for i in range(iterations):
        y, cache = propagate(X, parameters)
        it_cost = cost_function(y, Y)
        gradient = back_propagation(X, Y, parameters, cache)
        parameters = update(gradient, parameters, learning_rate)
        nn_cost.append(it_cost)
    return parameters, nn_cost
