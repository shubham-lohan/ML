# assume that layer_sizes tells your model about the input and
# output dimension too. However, that "might"
# not be the case with popular machine learning libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier


class MyNeuralNetwork():
    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation  # activation function
        self.learning_rate = learning_rate  # alpha
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_init = weight_init  # zero, random, normal

        self.history = []
        self.X_test_prob = []
        self.initialize_weights()

    def initialize_weights(self):
        shape = [(x, y)
                 for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        if self.weight_init == 'zero':
            self.weight = [self.zero_init(shapes) for shapes in shape]
        elif self.weight_init == 'random':
            self.weight = [self.random_init(shapes) for shapes in shape]
        else:
            self.weight = [self.normal_init(shapes) for shapes in shape]
        self.weightDiff = [None for i in range(self.n_layers - 1)]
        self.biasDiff = [None for i in range(self.n_layers - 1)]
        self.bias = [np.zeros(shape=(1, y)) for y in self.layer_sizes[1:]]
        self.X = [None for i in range(self.n_layers)]

    def sigmoid(self, z):
        return np.where(np.exp(-z) == np.inf, 0, 1/(1+np.exp(-z)))

    def sigmoid_grad(self, X):
        return X * (1.0 - X)

    def softmax(self, X):
        X -= np.max(X)
        y = np.exp(X)
        return (y / (y.sum(axis=1, keepdims=True)+1e-10))+1e-10

    def zero_init(self, shape):
        return np.zeros(shape)

    def random_init(self, shape):
        return np.random.rand(*shape)*0.01

    def normal_init(self, shape):
        print(shape, "shape")
        temp = np.random.normal(0, 1, size=shape)*0.01
        temp = np.reshape(temp, (shape))

        return temp

    def cross_entropy_loss(self, X, y):
        p = np.choose(y.reshape(-1, ), X)
        loss = -np.sum(np.log(p)) / len(p)
        return loss

    def fit(self, X, y, X_test, y_test, X_validate=[], y_validate=[]):

        batch_count = X.shape[0]//self.batch_size
        error = []
        test_error = []
        validation_error = []
        X_test_prob = []
        train_cost = []
        test_cost = []
        validation_cost = []
        for i in range(self.num_epochs):

            y_hat = self.predict(X)
            y_hat_test = self.predict(X_test)
            # y_hat_validate = self.predict(X_validate)
            test_cost.append(self.cross_entropy(y_hat_test, y_test))
            train_cost.append(self.cross_entropy(y_hat, y))
            # validation_cost.append(
            #     self.cross_entropy(y_hat_validate, y_validate))

            error.append(1 - self.score(X, y))
            test_error.append(1 - self.score(X_test, y_test))
            # validation_error.append(1-self.score(X_validate, y_validate))

            if i % 10 == 0:
                print(f"Iteration %d Training Acc.: %.10f Testing Acc.: %.10f\n" % (
                    i, 1-error[-1], 1-test_error[-1]))
                # print(f"Iteration %d Training Acc.: %.10f Validation Acc.: %.10f Testing Acc.: %.10f\n" % (
                #     i, 1-error[-1], 1-validation_error[-1], 1-test_error[-1]))

            for batch in range(batch_count):
                X_batch = X[self.batch_size *
                            batch: self.batch_size * (batch + 1), :]
                train_output = y[self.batch_size *
                                 batch: self.batch_size * (batch + 1)]

                temp_layer = np.zeros((X_batch.shape[0], 10))

                for k in range(self.batch_size):
                    temp_layer[k, int(train_output[k])] = 1

                self.history = temp_layer
                self.forward_propagation(X_batch)
                X_test_prob.append(self.predict_proba())
                self.back_propagation()
                for layer in range(self.n_layers - 1):
                    self.weight[layer] -= (self.learning_rate /
                                           self.batch_size) * self.weightDiff[layer]
                    self.bias[layer] -= (self.learning_rate/self.batch_size) * \
                        self.biasDiff[layer].sum(axis=0)

        self.X_test_prob = X_test_prob
        # return train_cost, test_cost
        self.train_cost, self.validation_cost, self.test_cost = train_cost, validation_cost, test_cost
        return self

    def activation_func(self, X):
        if self.activation == 'relu':
            return np.where(X > 0, X, 0)
        if self.activation == 'sigmoid':
            return np.where(np.exp(-X) == np.inf, 0, 1/(1+np.exp(-X)))
        if self.activation == 'linear':
            return X
        if self.activation == 'leaky_relu':
            return np.where(X >= 0, X, X * 0.01)
        if self.activation == 'tanh':
            return np.tanh(X)
        # softmax
        X -= np.max(X)
        y = np.exp(X)
        return y / y.sum(axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.X[0] = X
        for i in range(self.n_layers - 1):
            flag = False
            temp = np.dot(self.X[i], self.weight[i]) + self.bias[i]
            if i != self.n_layers-2:
                self.X[i+1] = self.activation_func(temp)
                flag = True
            elif not flag:
                self.X[i+1] = self.softmax(temp)
        return self.X

    def gradient_func(self, X):
        if self.activation == 'relu':
            return np.greater(X, 0).astype(int)
        if self.activation == 'sigmoid':
            return self.sigmoid_grad(X)
        if self.activation == 'linear':
            return np.ones(np.shape(X))
        if self.activation == 'leaky_relu':
            return np.where(X >= 0, 1, 0.01)
        if self.activation == 'tanh':
            return 1 - X**2
        else:
            return X*(1-X)  # softmax

    def back_propagation(self):
        for i in range(self.n_layers - 1, 0, -1):
            if i != self.n_layers - 1:
                temp = self.X[i]
                temp = self.gradient_func(temp)
                temp2 = (self.biasDiff[i].dot(self.weight[i].T)) * temp
                self.weightDiff[i-1] = (self.X[i-1].T).dot(temp2)
                self.biasDiff[i-1] = temp2
            else:
                error = self.predict_proba() - self.history
                # print(error, "errror here")
                self.weightDiff[i-1] = (self.X[i-1].T).dot(error)
                self.biasDiff[i-1] = error

    def predict_proba(self):
        return self.X[-1]

    def predict(self, X):
        self.X[0] = X
        for i in range(self.n_layers - 1):
            output_val = self.X[i] @ self.weight[i]
            if i == self.n_layers - 2:  # last layer
                output_val += self.bias[i]
                output_val = self.softmax(output_val)

            else:

                output_val += self.bias[i]
                output_val = self.activation_func(output_val)
            self.X[i+1] = output_val
        return np.argmax(self.predict_proba(), axis=1).T

    def score(self, X, y):
        prediction = self.predict(X) - y
        temp = np.where(prediction == 0)
        score = len(temp[0])/X.shape[0]
        return score

    def cross_entropy(self, y_hat, y):
        sum_score = 1e-9
        for i in range(len(y)):
            sum_score += y[i] * np.log(1e-15 + y_hat[i])
        mean_sum = 1.0 / len(y) * sum_score
        return -mean_sum

    def plot_graph(self):
        plt.plot(np.arange(self.num_epochs), self.train_cost,
                 color="green", label='Train Cost')
        # plt.plot(np.arange(self.num_epochs), self.test_cost,
        #          color="red", label='Test Cost')
        plt.plot(np.arange(self.num_epochs), self.test_cost,
                 color="blue", label='Validation Cost')
        plt.title(self.activation)
        plt.xlabel('Iterations')
        plt.ylabel('error')
        plt.legend()
        plt.savefig("plots/"+self.activation+".png")
        # plt.show()

    def save_weights(self):
        filename = 'weights/'+self.activation+"_model"
        pickle.dump(self, open(filename, 'wb'))


np.random.seed(0)


def pre_process():

    train_df = pd.read_csv('dataset/mnist_train.csv')
    test_df = pd.read_csv('dataset/mnist_test.csv')

    df = pd.concat([train_df, test_df])
    # print(df.columns)
    df = df.sample(frac=0.05).reset_index(drop=True)
    print(df.shape)
    df = df.to_numpy()

    n = df.shape[0]
    y = df[:, 0]
    df = df[:, 1:]/255
    df[df == 0] = 1e-10
    X_train = df[:int(n*0.7), :]
    y_train = y[:int(n*0.7)]

    # print(df)
    X_test = df[int(n*0.7):-int(n*0.2), :]
    y_test = y[int(n*0.7):-int(n*0.2)]

    X_validate = df[-int(n*0.2):, 1:]
    y_validate = y[-int(n*0.2)]

    print(X_train.shape, y_train.shape, X_validate.shape,
          y_validate.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_validate, y_validate, X_test, y_test

    # return X_train[:2000, :], y_train[:2000], X_validate[:500, :], y_validate[:500], X_test[:700, :], y_test[:700]


def load_weight(filename):
    filename = "weights/"+filename+"_model"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


if __name__ == '__main__':
    X_train, y_train, X_validate, y_validate, X_test, y_test = pre_process()
    # print(X_train.shape, y_train.shape, X_validate.shape,
    #       y_validate.shape, X_test.shape, y_test.shape)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    batch_size = len(X_train)//10
    layers = [784, 256, 128, 64, 32, 10]
    learning_rate = 0.08
    wt_initializer = 'normal'
    epoch = 150
    # 1

    # nn_relu = MyNeuralNetwork(len(layers), layers,
    #                           'relu', learning_rate, wt_initializer, batch_size, 150)
    # nn_relu.fit(X_train, y_train, X_test, y_test)
    # print(nn_relu.score(X_test, y_test))
    # nn_relu.plot_graph()
    # nn_relu.save_weights()
    # 0.09523809523809523

    # 2

    # nn_leaky_relu = MyNeuralNetwork(len(layers), layers,
    #                           'leaky_relu', learning_rate, wt_initializer, batch_size, 150)
    # nn_leaky_relu.fit(X_train, y_train, X_test, y_test)
    # print(nn_leaky_relu.score(X_test, y_test))
    # nn_leaky_relu.plot_graph()
    # nn_leaky_relu.save_weights()
    # 0.0952380952

    # 3

    # nn_sigmoid = MyNeuralNetwork(len(layers), layers,
    #                           'sigmoid', learning_rate, wt_initializer, batch_size, 150)
    # nn_sigmoid.fit(X_train, y_train, X_test, y_test)
    # print(nn_sigmoid.score(X_test, y_test))
    # nn_sigmoid.plot_graph()
    # nn_sigmoid.save_weights()

    # 4

    # nn_tanh = MyNeuralNetwork(len(layers), layers,
    #                           'tanh', learning_rate, wt_initializer, batch_size, 150)
    # nn_tanh.fit(X_train, y_train, X_test, y_test)
    # print(nn_tanh.score(X_test, y_test))
    # nn_tanh.plot_graph()
    # nn_tanh.save_weights()

    # 5
    # nn_linear = MyNeuralNetwork(len(layers), layers,
    #                           'linear', learning_rate, wt_initializer, batch_size, 150)
    # nn_linear.fit(X_train, y_train, X_test, y_test)
    # print(nn_linear.score(X_test, y_test))
    # nn_linear.plot_graph()
    # nn_linear.save_weights()

    # 6
    # nn_softmax = MyNeuralNetwork(len(layers), layers,
    #                           'softmax', learning_rate, wt_initializer, batch_size, 150)
    # nn_softmax.fit(X_train, y_train, X_test, y_test)
    # print(nn_softmax.score(X_test, y_test))
    # nn_softmax.plot_graph()
    # nn_softmax.save_weights()

    # using sklearn

    # nn_relu = MLPClassifier(hidden_layer_sizes=layers[1:-1], solver='sgd', activation='relu',
    #                         learning_rate='constant', learning_rate_init=learning_rate, max_iter=epoch)
    # nn_relu.fit(X_train, y_train)
    # train_acc = nn_relu.score(X_train, y_train)
    # test_acc = nn_relu.score(X_test, y_test)
    # print("ReLU")
    # print('Testing accuracy:', test_acc)

    # nn_linear = MLPClassifier(hidden_layer_sizes=layers[1:-1], solver='sgd', activation='identity',
    #                           learning_rate='constant', learning_rate_init=learning_rate, max_iter=epoch)
    # nn_linear.fit(X_train, y_train)
    # train_acc = nn_linear.score(X_train, y_train)
    # test_acc = nn_linear.score(X_test, y_test)
    # print("Linear")
    # print('Testing accuracy:', test_acc)

    # nn_sigmoid = MLPClassifier(hidden_layer_sizes=layers[1:-1], solver='sgd', activation='logistic',
    #                            learning_rate='constant', learning_rate_init=learning_rate, max_iter=epoch)
    # nn_sigmoid.fit(X_train, y_train)
    # train_acc = nn_sigmoid.score(X_train, y_train)
    # test_acc = nn_sigmoid.score(X_test, y_test)
    # print("Sigmoid")
    # print('Testing accuracy:', test_acc)

    # nn_tanh = MLPClassifier(hidden_layer_sizes=layers[1:-1], solver='sgd', activation='tanh',
    #                         learning_rate='constant', learning_rate_init=learning_rate, max_iter=epoch)
    # nn_tanh.fit(X_train, y_train)
    # train_acc = nn_tanh.score(X_train, y_train)
    # test_acc = nn_tanh.score(X_test, y_test)
    # print("tanh")
    # print('Testing accuracy:', test_acc)

    """
    ReLU
    Testing accuracy: 0.9257142857142857
    Linear
    Testing accuracy: 0.09428571428571429
    Sigmoid
    Testing accuracy: 0.10285714285714286
    tanh
    Testing accuracy: 0.9114285714285715
    """

    # 5.

    learning_rates = [0.001, 0.01, 0.1, 1]
    for i in learning_rates:
        model = MLPClassifier(hidden_layer_sizes=layers[1:-1], solver='sgd', activation='tanh',learning_rate='constant', learning_rate_init=i, max_iter=epoch)
        model.fit(X_train, y_train)
        testing_accuracy = model.score(X_test, y_test)
        print("Testing Accuracy for learning rate =", i,
              "and activation function used tanh", ": ", testing_accuracy)
