# Neural Network -Python
Creating a neural network with two hidden layers

# Activation functions

File - activation.py - contains different activation function, that can be used by neurons.
![image](https://user-images.githubusercontent.com/37414943/60324416-08de5780-9985-11e9-8014-91fd3e921601.png)

# Creating a neural network

File - neural.py - contains functions that are related to neural network itself. Using theese functions one is able to initialize a network,
calculate networks response (forward propagation) and cost function. 
![image](https://user-images.githubusercontent.com/37414943/60324567-6e324880-9985-11e9-953d-f37e0b564668.png)

To updates its weights and biases network uses back propagation method.
![image](https://user-images.githubusercontent.com/37414943/60324646-a043aa80-9985-11e9-9947-2a567b5e27ee.png)

# Training process

All functions, that have been described above, are then used in train function, witch is responsible for training a given neural network,
based on parameters defined by the user.
![image](https://user-images.githubusercontent.com/37414943/60324756-ec8eea80-9985-11e9-8d0d-957a59a50e6a.png)

Using sklearn we create a data set on witch our network will work. Afterwords we train the net and plot it's learning progess.
![image](https://user-images.githubusercontent.com/37414943/60324849-2eb82c00-9986-11e9-9205-28bb8d776901.png)

# Cost function
Learning process is presented by drawing a chart representing changes in cost function during networks learning process.
![image](https://user-images.githubusercontent.com/37414943/60324962-6a52f600-9986-11e9-8811-4e2323112319.png)

As we can see thanks to back propagation our neural network quickly converges to a minimum.
