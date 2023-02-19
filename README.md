This is a Python implementation of a simple feedforward neural network using the backpropagation algorithm. 
To show a more understanding the network is designed to for a binary classifiction.
Net work will classify individuals as either "obese" or "not obese" based on their weight and height using a threshold BMI value.

**Usage**\\
To use the code, simply clone the repository and run the neural_network.py file. The program generates a random dataset of 300 individuals with their weights and heights, and then trains a neural network to classify them as "obese" or "not obese" based on a threshold BMI value of 25.

The example neural network has a 2-4-1 architecture, meaning it has two input nodes, four hidden nodes, and one output node. The activation function used in the hidden layer is the hyperbolic tangent (tanh) function, and the output layer uses a linear activation function.

The program outputs the total cost of the network after training on the entire dataset. The cost is defined as the sum of the squared errors between the predicted and actual output values, divided by two.

**License**\\
This project is licensed under the MIT License
