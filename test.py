import numpy as np
import matplotlib.pyplot as plt


# Class to create a neural network with single neuron class NeuralNetwork (object):
class NeuralNetwork(object):
    def __init__(self, num_params=2):
        # Using seed to make sure it'll generate same weights in every run
        # np.random.seed(1)
        # 3x1 Weight matrix
        self.weight_matrix = (
            2 * np.random.random((num_params + 1, 1)) - 1
        )  # random weights between -1 and 1

        self.rate = 1

    # hard_limiter as activation fucntion
    def hard_limiter(self, x):
        outs = np.zeros(x.shape)
        outs[x > 0] = 1
        return outs

    # forward propagation
    def forward_propagation(self, inputs):
        outs = np.dot(inputs, self.weight_matrix)
        return self.hard_limiter(outs)

    # predicting the classes of new data points
    def pred(self, inputs):
        preds = self.forward_propagation(inputs)
        return preds

    # training the neural network.
    def train(self, train_inputs, train_outputs, num_train_iterations=10):
        # Number of iterations we want to perform for this set of input.
        for iteration in range(num_train_iterations):
            # updating the perceptron base on the misclassified examples
            for i in range(train_inputs.shape[0]):
                pred_i = self.pred(train_inputs[i, :])
                if pred_i != train_outputs[i]:
                    output = self.forward_propagation(train_inputs[i, :])
                    # Calculate the error in the output.
                    error = train_outputs[i] - output
                    adjustment = self.rate * error * train_inputs[i]
                    # Adjust the weight matrix
                    self.weight_matrix[:, 0] += adjustment
                    # plot the seperating line based on the weights
            print("Iteration #" + str(iteration))
            plot_fun_thr(
                train_inputs[:, 1:3], train_outputs, self.weight_matrix[:, 0], classes
            )


def plot_fun(features, labels, classes):
    # ploting the data points
    plt.plot(
        features[labels[:] == classes[0], 0],
        features[labels[:] == classes[0], 1],
        "rs",
        features[labels[:] == classes[1], 0],
        features[labels[:] == classes[1], 1],
        "g^",
    )
    plt.axis([-1, 2, -1, 2])
    plt.xlabel("x: feature 1")
    plt.ylabel("y: feature 2")
    plt.legend(["Class" + str(classes[0]), "Class" + str(classes[1])])
    plt.grid()
    plt.show()


def plot_fun_thr(features, labels, thre_parms, classes):
    # ploting the data points
    plt.plot(
        features[labels[:] == classes[0], 0],
        features[labels[:] == classes[0], 1],
        "rs",
        features[labels[:] == classes[1], 0],
        features[labels[:] == classes[1], 1],
        "g^",
    )
    plt.axis([-1, 2, -1, 2])
    # ploting the seperating line
    x1 = np.linspace(-1, 2, 50)
    x2 = (
        -(thre_parms[1] * x1 + thre_parms[0]) / thre_parms[2]
    )  # a X1 + b X2 + c=0 --> x2 = -(a X1 + c)/b
    plt.plot(x1, x2, "-r")
    plt.xlabel("x: feature 1")
    plt.ylabel("y: feature 2")
    plt.legend(["class" + str(classes[0]), "class" + str(classes[1])])
    # plt.pause(0.5)
    plt.show()


features = np.array([[1, 0], [0, 1], [-1, 0], [-1, 1], [0, -1]])
print(features)
labels = np.array([1, 1, 0, 0, 0])
print(labels)
classes = [0, 1]

plot_fun(features, labels, classes)


bias = np.ones((features.shape[0], 1))  # expanding the feature space by adding
# the bias vector
print("Our bias:")
print(bias)
print("bias shape:")
print(bias.shape)
featuresNew = np.append(bias, features, axis=1)
print("Features vector after adding the bias")
print(featuresNew)
print(featuresNew.shape)

neural_network = NeuralNetwork()
print("Random weights at the start of training")
print(neural_network.weight_matrix)

num_iterations = 100
neural_network.train(featuresNew, labels, num_iterations)
print("New weights after training")
print(neural_network.weight_matrix)

# Test the neural network with training data points.
print("Testing network on training data points ->")
print(neural_network.pred(featuresNew))

# e) Using the trained perceptron, classify the test data samples given in the table below by calling the pred() function.
# def pred(self,inputs)

Test = np.array([[2, 2], [-2, -2], [0, 0], [-2, 0]])
bias = np.ones((Test.shape[0], 1))
print("Our bias:")
print(bias)
print("Our Test Sample:")
print(Test)
print("We are going to test:")
Testnew = np.append(bias, Test, axis=1)
print(Testnew)

print("Classifying the test smaples given(predicted labels):")
neural_network.pred(Testnew)
