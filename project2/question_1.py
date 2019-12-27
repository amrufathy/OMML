#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from cvxopt import matrix, solvers
from data_extraction import load_mnist, SEED
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import shuffle


data_path = os.path.join(os.getcwd(), 'Data')

x_train24, y_train24, x_test24, y_test24 = load_mnist(data_path, kind='train')


# # functions

def train_val(k, x, y):
    idx = [n for n in range(len(x)) if n != k]
    x_test = x[k]
    y_test = y[k]
    x_train = np.empty([0, x[1].shape[1]])
    y_train = np.empty([0, ])
    for i in idx:
        x_train = np.concatenate((x_train, x[i]))
        y_train = np.concatenate((y_train, y[i]))
    return x_train, y_train, x_test, y_test


# C, x, y, kernel function
def QP(x, y, rbf_gamma, C):
    """
    Solving SVM nonlinear dual soft:
    min(lambda) = lambdaT * Q * lambda + PT * lambda
    inequalities: G * lambda <= h ==> lambda <= C and -lambda <= 0
    equalities: A * lambda = b ==> lambdaT * y = 0
    """
    # In QP formulation (dual): n variables, 2n+1 constraints (1 equation, 2n inequations)
    n = x.shape[0]  # number of samples in data
    # print('x.shape: ', x.shape, 'y.shape: ', y.shape)
    Q = kernel_rbf_Q(x, y, rbf_gamma)  # write function for kernal
    Q = matrix(Q, tc='d')  # transforming for solver (n,n)
    e = np.ones((n, 1))  # (n,1)
    P = matrix(-e, tc='d')  # (n,1)
    # equalities 
    A = matrix(y.reshape((1, -1)), tc='d')  # (#of equalities,n)
    b = matrix([0.0])  # (#of equalities*1)=(1,1)
    # inequalities
    # 2 inequalities for each sample -lambda <= 0 and lambda <= C
    h = matrix(np.r_[np.zeros((n, 1)), np.zeros((n, 1)) + C], tc='d')  # (2*n,1)
    G = matrix(np.r_[-np.eye(n), np.eye(n)], tc='d')  # (n*2,#of ineq for each sample=2)
    # print('Q: ', Q.size)
    # print('P: ', P.size)
    # print('G: ', G.size)
    # print('h: ', h.size)
    # print('A: ', A.size)
    # print('b: ', b.size)
    solvers.options['show_progress'] = False
    solution = solvers.qp(Q, P, G, h, A, b)
    if solution['status'] != 'optimal':
        print('Not PSD!')
    else:
        return solution


def kernel_rbf_Q(x, y, rbf_gamma):
    P = y.shape[0]
    Q = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            Q[i, j] = y[i] * y[j] * np.exp(-rbf_gamma * np.linalg.norm(x[i] - x[j]) ** 2)
    return Q


def kernel_rbf_elem(x, y, rbf_gamma):
    K = rbf_kernel(X=x, Y=y, gamma=rbf_gamma)
    return K


def prediction_full_P(x, y, lambda_vector, b, x_test, rbf_gamma):
    predictions_y = np.zeros((x_test.shape[0],))
    for t in range(x_test.shape[0]):
        decision_func_val = 0
        for i in range(x.shape[0]):
            k = kernel_rbf_elem(x[i].reshape(1, -1), x_test[t].reshape(1, -1), rbf_gamma)
            decision_func_val = decision_func_val + (lambda_vector[i][0] * y[i] * k)
        predictions_y[t] = np.sign(decision_func_val + b)
    return predictions_y


def optimal_b_star(x, y, lambda_vector, rbf_gamma):
    """
    Whenever lambda optimal is strictly positive,
    b should be a unique value for all the training samples
    """
    sum_mult = 0
    i = np.argmax(lambda_vector)  # index of nonzero element of lambda with max gap
    for j in range(x.shape[0]):
        k = kernel_rbf_elem(x[j].reshape(1, -1), x[i].reshape(1, -1), rbf_gamma)
        mult = lambda_vector[j][0] * y[j] * k
        sum_mult = sum_mult + mult
    b = (1 - y[i] * sum_mult)[0][0] / y[i]
    return b


def SVM_prediction(x_train, y_train, rbf_gamma, C, x_test, y_test):
    result = []
    QPsolution = QP(x_train, y_train, rbf_gamma, C)
    # print('QPsolution:', QPsolution)
    lambda_vector = np.array(QPsolution['x'])
    # filtering lambda vector is striclty positive (1e-7 is counted as 0)
    lambda_vector[lambda_vector < 1e-7] = 0
    print('number of nonzero lambda: ', np.count_nonzero(lambda_vector))
    b = optimal_b_star(x_train, y_train, lambda_vector, rbf_gamma)
    y_pred = prediction_full_P(x_train, y_train, lambda_vector, b, x_test, rbf_gamma)
    result.extend((QPsolution, y_pred))
    return result


# # main

# best HPs:
gamma = 0.001  # for rbf kernel
c = 100
res_test = SVM_prediction(x_train24, y_train24, gamma, c, x_test24, y_test24)
acc_test = accuracy_score(y_test24, res_test[1])
conf_matrix = confusion_matrix(y_test24, res_test[1])  # first row tn, fp; second row fn, tp
print('TEST INFO')
print(c, gamma, acc_test)
print(conf_matrix)

res_train = SVM_prediction(x_train24, y_train24, gamma, c, x_train24, y_train24)
acc_train = accuracy_score(y_train24, res_train[1])
conf_matrix = confusion_matrix(y_train24, res_train[1])
print('TRAIN INFO')
print(c, gamma, acc_train)
print(conf_matrix)

# # K-fold (IF YOU RUN IT, IT TAKES YEARS)

k_folds = 5
x_train24, y_train24 = shuffle(x_train24, y_train24, random_state=SEED)
y_train24_sub = np.split(y_train24, k_folds)
x_train24_sub = np.split(x_train24, k_folds)

C_params = [0.01, 0.1, 1, 10, 100, 1000, 10000]
Gamma_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

result_dict = dict()
for c in C_params:
    for gamma in Gamma_params:
        param_key = str(c) + '__' + str(gamma)
        print(param_key)
        # one pair of parameters
        result_k_fold = []
        for k in range(k_folds):
            x_train, y_train, x_test, y_test = train_val(2, x_train24_sub, y_train24_sub)
            res = SVM_prediction(x_train, y_train, gamma, c, x_test, y_test)
            result_k_fold.append(res)
            acc = accuracy_score(y_test, res[1])
            if res[1] < 0.6:
                break
        result_dict.update({param_key: result_k_fold})

print(result_dict)
