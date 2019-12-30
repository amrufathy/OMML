#!/usr/bin/env python
# coding: utf-8

import os
import time

import numpy as np
from cvxopt import matrix, solvers
from data_extraction import load_mnist
from sklearn.metrics.pairwise import rbf_kernel

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


def KKT_violation(lambda_, train_y, C):
    e = -1. * np.ones(shape=(train_y.shape[0], 1))
    train_y_ = train_y.reshape(len(train_y), 1)
    gradient = np.copy(e)
    grad_y = -gradient / train_y_
    idx = np.arange(0, len(train_y_)).reshape((len(train_y_), 1))
    epsillon = 1e-7
    lambda_l = lambda_ <= epsillon
    lambda_u = np.logical_and(lambda_ >= (C - epsillon), lambda_ <= C)

    l_plus_cond, l_minus_cond = np.logical_and(
        lambda_l, train_y_ == 1), np.logical_and(lambda_l, train_y_ == -1)
    l_plus, l_minus = list(idx[l_plus_cond]), list(idx[l_minus_cond])

    u_plus_cond, u_minus_cond = np.logical_and(
        lambda_u, train_y_ == 1), np.logical_and(lambda_u, train_y_ == -1)
    u_plus, u_minus = list(idx[u_plus_cond]), list(idx[u_minus_cond])

    f_cond = np.logical_and(lambda_ > epsillon,
                            lambda_ < (C - epsillon))
    f = list(idx[f_cond])

    R, S = sorted(l_plus + u_minus + f), sorted(l_minus + u_plus + f),

    m_lambda, M_lambda = round(
        grad_y[R].min(), 10), round(grad_y[S].min(), 10)
    return m_lambda - M_lambda


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


def SVM_prediction(x_train, y_train, rbf_gamma, C, x_test, y_test, test=True):
    result = []
    tik = time.time()
    QPsolution = QP(x_train, y_train, rbf_gamma, C)
    tok = time.time()
    computational_time = tok - tik
    # if test==True:
    # file = open('question_1.log', mode='w+')
    # print(f'QP COMPUTATIONAL TIME: {} SEC', file=file)
    # file.close()
    # print('QPsolution:', QPsolution)
    lambda_vector = np.array(QPsolution['x'])
    # filtering lambda vector is striclty positive (1e-7 is counted as 0)
    lambda_vector[lambda_vector < 1e-7] = 0
    # print('number of nonzero lambda: ', np.count_nonzero(lambda_vector))
    b = optimal_b_star(x_train, y_train, lambda_vector, rbf_gamma)
    y_pred = prediction_full_P(x_train, y_train, lambda_vector, b, x_test, rbf_gamma)
    result.extend((QPsolution, y_pred, np.round(computational_time, 2)))
    return result
