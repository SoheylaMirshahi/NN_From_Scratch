import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


class Layer:
    def __init__(self, n_nodeIn, n_nodeOut):
        self.costGredientW = np.zeros((n_nodeIn, n_nodeOut))
        self.costGredientB = np.zeros((n_nodeOut,))
        self.n_nodeIn = n_nodeIn
        self.n_nodeOut = n_nodeOut
        self.weights = np.random.randn(n_nodeIn, n_nodeOut) / np.sqrt(n_nodeIn)
        self.biases = np.random.randn(n_nodeOut)

    def apply_gradients(self, learning_rate):
        self.biases -= self.costGredientB * learning_rate
        self.weights -= self.costGredientW * learning_rate

    def activation_function_tanh(self, x):
        return np.tanh(x)

    def calculate_outputs(self, items):
        output = np.dot(items, self.weights) + self.biases
        return self.activation_function_tanh(output)

    def node_cost(self, computed_output, actual_output):
        error = computed_output - actual_output
        return error ** 2 / 2


class NeuralNetwork:

    def __init__(self, layer_size):
        self.layers = [Layer(layer_size[i], layer_size[i + 1]) for i in range(len(layer_size) - 1)]

    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def cost(self, datapoint):
        inputs = datapoint[:2]
        expected_output = datapoint[2]
        output = self.calculate_outputs(inputs)
        cost = self.layers[-1].node_cost(output, expected_output)
        return cost

    def cost_for_all_data(self, data):
        total_cost = 0
        for datapoint in data:
            total_cost += self.cost(datapoint)
        return total_cost / len(data)

    def learning(self, training_data, learning_rate):
        for layer in self.layers:
            layer.costGredientW.fill(0)
            layer.costGredientB.fill(0)
        for datapoint in training_data:
            inputs = datapoint[:2]
            expected_output = datapoint[2]

            # forward pass
            outputs = [inputs]
            for layer in self.layers:
                output = layer.calculate_outputs(outputs[-1])
                outputs.append(output)

            # backward pass
            cost_gradient = (outputs[-1] - expected_output) * (1 - outputs[-1] ** 2)
            for layer_index in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_index]
                layer.costGredientB += cost_gradient
                layer.costGredientW += np.outer(outputs[layer_index], cost_gradient)
                cost_gradient = np.dot(cost_gradient, layer.weights.T) * (1 - outputs[layer_index] ** 2)

        for layer in self.layers:
            layer.apply_gradients(learning_rate)


def calculate_bmi(weight, height):
    return weight / (height / 100) ** 2


def classify_fatness(bmi):
    return 1 if bmi >= 25 else 0


weights = np.random.randint(50, 100, 300)
heights = np.random.randint(150, 200, 300)

bmis = [calculate_bmi(weights[i], heights[i]) for i in range(300)]
fatness = [classify_fatness(bmis[i]) for i in range(300)]

data = np.column_stack((weights, heights, fatness))

network = NeuralNetwork([2, 4, 1])
total_cost = network.cost_for_all_data(data)
print("Total cost:", total_cost)