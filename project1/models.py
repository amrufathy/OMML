import itertools
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

SEED = 1848399
random.seed(SEED)
np.random.seed(SEED)


class Network:
    def __init__(self, hidden_size, input_size, output_size, _rho):
        # save to use later
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rho = _rho

    def forward(self, *args):
        raise NotImplementedError

    def loss(self, omega, inputs, labels):
        # calculate loss
        outputs = self.forward(inputs, omega)
        error = np.mean(np.square(outputs - labels))
        regularization = self.rho * np.square(np.linalg.norm(omega))

        return error + regularization

    def fit(self, *args):
        raise NotImplementedError

    def extreme_learning(self, *args):
        raise NotImplementedError

    def decomposition(self, *args):
        raise NotImplementedError

    def save(self, *args):
        raise NotImplementedError

    def load(self, *args):
        raise NotImplementedError

    def surface_plot(self, inputs, optimal_params, title=''):
        # TODO (Amr): not final yet, but working so far
        outputs = self.forward(inputs, optimal_params)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # noinspection PyPep8Naming
        X1, X2, Y = inputs[:, 0], inputs[:, 1], outputs.ravel()
        ax.scatter(X1, X2, Y, color='red', alpha=1)
        ax.plot_trisurf(X1, X2, Y, cmap='viridis', edgecolor='none')

        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        plt.show()


class MLP(Network):
    def __init__(self, hidden_size, input_size=2, output_size=1, _rho=1e-5):
        # initialize weights and biases
        self.W = np.random.rand(input_size, hidden_size)
        self.V = np.random.rand(hidden_size, output_size)
        self.b = np.random.rand(1, hidden_size)

        super().__init__(hidden_size, input_size, output_size, _rho)

    def forward(self, inputs, omega):
        self.__unpack_omega(omega)
        intermediate_output = np.tanh(np.dot(inputs, self.W) - self.b)
        return np.dot(intermediate_output, self.V)

    def fit(self, inputs, labels):
        # omega contains all free params of the network
        omega = np.concatenate([self.V, self.W.reshape(self.W.size, 1), self.b.T])
        self.__run_minimization(inputs, labels, omega)

    def extreme_learning(self, inputs, labels):
        # omega contains `V` only
        omega = self.V
        self.__run_minimization(inputs, labels, omega)

    def decomposition(self, inputs, labels):
        raise NotImplementedError('Decomposition method is not implemented for the MLP network!')

    def test_loss(self, inputs, labels):
        # only for use on val/test data, not during training
        omega = np.concatenate([self.V, self.W.reshape(self.W.size, 1), self.b.T])
        outputs = self.forward(inputs, omega)
        return np.mean(np.square(outputs - labels))

    def __run_minimization(self, inputs, labels, omega):
        # initial error
        print(f'Initial training error: {self.test_loss(inputs, labels):.4f}')
        print(f'Initial value of objective function: {self.loss(omega, inputs, labels):.4f}')
        # back-propagation
        tik = time.time()
        optimal = optimize.minimize(fun=self.loss, x0=omega, args=(inputs, labels))
        tok = time.time()

        # print out required info
        self.__print_training_info(inputs, labels, optimal, tok - tik)

    def __print_training_info(self, inputs, labels, result, elapsed_time):
        print(f'Number of neurons: {self.hidden_size}')
        print(f'Value of sigma: {1}')
        print(f'Value of rho: {self.rho}')
        print(f'Solver: BFGS (Default)')
        print(f'Final value of objective function: {result.fun:.4f}')
        print(f'Final value of gradient: {np.linalg.norm(result.jac):.4f}')
        print(f'Number of iterations: {result.nit}')
        print(f'Number of function evaluations: {result.nfev}')
        print(f'Number of gradient evaluations: {result.njev}')
        print(f'Time for optimization: {elapsed_time:.4f} seconds')
        print(f'Termination message: {result.message}')
        print(f'Final Training error: {self.test_loss(inputs, labels):.4f}')

    def save(self, filename=''):
        omega = np.concatenate([self.V, self.W.reshape(self.W.size, 1), self.b.T])
        filename = 'mlp_weights' if filename == '' else filename
        np.save(filename, omega)

    def load(self, filename=''):
        filename = 'mlp_weights.npy' if filename == '' else filename
        omega = np.load(filename)
        self.__unpack_omega(omega)

    def surface_plot(self, inputs, title='', *args):
        optimal_parameters = np.concatenate([self.V, self.W.reshape(self.W.size, 1), self.b.T])
        super().surface_plot(inputs, optimal_parameters, 'MLP' if title == '' else title)

    def __unpack_omega(self, omega):
        # `V` always exists
        self.V = omega[:self.V.size].reshape(*self.V.shape)
        # if `W`, `b` are in omega, unpack it (full minimization)
        if omega.size > self.V.size:
            self.W = omega[self.V.size: self.V.size + self.W.size].reshape(*self.W.shape)
            self.b = omega[self.V.size + self.W.size:].reshape(*self.b.shape)


class RBF(Network):
    def __init__(self, hidden_size, input_size=2, output_size=1, _rho=1e-4, _sigma=1.):
        # initialize weights and biases
        self.C = np.random.rand(input_size, hidden_size)
        self.V = np.random.rand(hidden_size, output_size)

        self.sigma = _sigma

        super().__init__(hidden_size, input_size, output_size, _rho)

    def forward(self, inputs, omega):
        self.__unpack_omega(omega)

        # C needs to be in shape (#samples, dim, #centroids)
        c = np.tile(self.C, (inputs.shape[0], 1, 1))

        intermediate_output = np.zeros((inputs.shape[0], self.hidden_size))

        # hidden units = #centroids
        for i in range(self.hidden_size):
            # subtract all points from centroid
            # take norm of each distance vector (axis = 1)
            intermediate_output[:, i] = self.gaussian(
                np.linalg.norm(inputs - c[:, :, i], axis=1))

        return np.dot(intermediate_output, self.V)

    def gaussian(self, z):
        return np.exp(-np.square(z / self.sigma))

    def fit(self, inputs, labels):
        # omega contains all free params of the network
        omega = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])
        self.__run_minimization(inputs, labels, omega)

    def extreme_learning(self, inputs, labels):
        # pick `N` centers from `inputs`
        self.C = np.array(random.choices(inputs, k=self.hidden_size)).T
        # omega contains `V` only
        omega = self.V
        self.__run_minimization(inputs, labels, omega)

    def decomposition(self, inputs, labels):
        tik = time.time()
        early_stopping_cond = 1e-5
        sum_of_gradients, i, max_iters = 1, 0, 50

        clusters = KMeans(n_clusters=self.hidden_size, random_state=SEED).fit(inputs)
        self.C = np.array(clusters.cluster_centers_).T

        omega = self.V

        print(f'Initial training error: {self.test_loss(inputs, labels):.4f}')
        print(f'Initial value of objective function: {self.loss(omega, inputs, labels):.4f}')

        while sum_of_gradients > early_stopping_cond and i < max_iters:
            # optimize V
            omega = self.V
            optimizer1 = self.__run_minimization(inputs, labels, omega)
            gradient_1 = np.linalg.norm(optimizer1.jac.T)
            self.V = optimizer1.x.reshape(*self.V.shape)

            # optimize C
            omega = self.C.reshape(self.C.size, 1)
            optimizer2 = self.__run_minimization(inputs, labels, omega)
            gradient_2 = np.linalg.norm(optimizer2.jac.T)
            self.C = optimizer2.x.reshape(*self.C.shape)

            sum_of_gradients = gradient_1 + gradient_2
            i += 1

        tok = time.time()

        self.__print_training_info(inputs, labels, optimizer2, tok - tik)

    def __run_minimization(self, inputs, labels, omega):
        # initial error
        print(f'Initial training error: {self.test_loss(inputs, labels):.4f}')
        print(f'Initial value of objective function: {self.loss(omega, inputs, labels):.4f}')
        # back-propagation
        tik = time.time()
        optimal = optimize.minimize(fun=self.loss, x0=omega, args=(inputs, labels))
        tok = time.time()

        # print out required info
        self.__print_training_info(inputs, labels, optimal, tok - tik)

        return optimal

    def __print_training_info(self, inputs, labels, result, elapsed_time):
        print(f'Number of neurons: {self.hidden_size}')
        print(f'Value of sigma: {1}')
        print(f'Value of rho: {self.rho}')
        print(f'Solver: BFGS (Default)')
        print(f'Final value of objective function: {result.fun:.4f}')
        print(f'Final value of gradient: {np.linalg.norm(result.jac):.4f}')
        print(f'Number of iterations: {result.nit}')
        print(f'Number of function evaluations: {result.nfev}')
        print(f'Number of gradient evaluations: {result.njev}')
        print(f'Time for optimization: {elapsed_time:.4f} seconds')
        print(f'Termination message: {result.message}')
        print(f'Final Training error: {self.test_loss(inputs, labels):.4f}')

    def test_loss(self, inputs, labels):
        # only for use on val/test data, not during training
        omega = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])
        outputs = self.forward(inputs, omega)
        return np.mean(np.square(outputs - labels))

    def save(self, filename=''):
        omega = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])
        filename = 'rbf_weights' if filename == '' else filename
        np.save(filename, omega)

    def load(self, filename=''):
        filename = 'rbf_weights.npy' if filename == '' else filename
        omega = np.load(filename)
        self.__unpack_omega(omega)

    def surface_plot(self, inputs, title='', *args):
        optimal_parameters = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])
        super().surface_plot(inputs, optimal_parameters, 'RBF' if title == '' else title)

    def __unpack_omega(self, omega):
        # check omega size
        if omega.size == self.V.size:
            self.V = omega[:self.V.size].reshape(*self.V.shape)
        elif omega.size == self.C.size:
            self.C = omega[:self.C.size].reshape(*self.C.shape)
        else:
            self.V = omega[:self.V.size].reshape(*self.V.shape)
            self.C = omega[self.V.size:].reshape(*self.C.shape)


def grid_search():
    # print everything to file
    sys.stdout = open('log_file.log', 'w+')

    N = [5, 10, 25, 50]  # hidden units
    rho = [1e-3, 1e-4, 1e-5]  # regularization weight
    sigma = [0.25, 0.5, 1, 2]  # spread of gaussian function (RBF)

    best_val_err = np.inf
    best_params = None

    print('=' * 10)
    print('MLP')
    print('=' * 10)

    for params in itertools.product(*(N, rho)):
        n, r = params
        mlp = MLP(hidden_size=n, _rho=r)
        mlp.fit(x_train, y_train)

        val_err = mlp.test_loss(x_val, y_val)
        test_err = mlp.test_loss(x_test, y_test)
        print(f'\nParams: {params} <=> Val error: {val_err:.4f}, Test error: {test_err:.4f}')

        if val_err < best_val_err:
            best_val_err = val_err
            best_params = params

        print('\n-------------\n')

    print(f'Best MLP params: {best_params}')

    best_val_err = np.inf
    best_params = None

    print('=' * 10)
    print('RBF')
    print('=' * 10)

    for params in itertools.product(*(N, rho, sigma)):
        n, r, s = params
        rbf = RBF(hidden_size=n, _rho=r, _sigma=s)
        rbf.fit(x_train, y_train)

        val_err = rbf.test_loss(x_val, y_val)
        test_err = rbf.test_loss(x_test, y_test)
        print(f'\nParams: {params} <=> Val error: {val_err:.4f}, Test error: {test_err:.4f}')

        if val_err < best_val_err:
            best_val_err = val_err
            best_params = params

        print('\n-------------\n')

    print(f'Best RBF params: {best_params}')


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'omml', 'project1', 'data_points.csv')
    dataset = np.genfromtxt('project1/data_points.csv', delimiter=',')

    x = dataset[1:, :2]
    y = dataset[1:, 2]
    y = np.expand_dims(y, -1)  # row -> column vector

    # train 70%, validation 15%, test 15%
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, train_size=0.7, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, train_size=0.5, random_state=SEED)

    # hyper-parameter tuning
    # grid_search()

    # Train & Save best params
    # Best params
    # MLP: (25, 1e-5)
    # RBF: (25, 1e-5, 1)

    mlp = MLP(hidden_size=25, _rho=1e-5)
    mlp.fit(x, y)
    mlp.save()
