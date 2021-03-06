import logging
import os
import time

import numpy as np
from cvxopt import matrix, solvers
from data_extraction import load_mnist
from utils import initialize_logger


class SVMDecomposition(object):
    def __init__(self, logging_path, C=2.5, gamma=0.01):
        initialize_logger(logging_path)
        self.C = C
        self.gamma = gamma
        self.logging_path = logging_path
        self.data_path = os.path.join(os.getcwd(), 'project2', 'Data')
        self.train_x, self.train_y, self.test_x, self.test_y = load_mnist(
            self.data_path, kind='train')
        logging.info('Dataset is loaded.')
        self.bias = 0
        self.hessian_mat = np.zeros(shape=(self.train_y.shape[0],
                                           self.train_y.shape[0]))
        self.e = -1. * np.ones(shape=(self.train_y.shape[0], 1))
        self.lambda_star = None

        print(f'C = {self.C}, gamma = {self.gamma}')

    def rbf_kernel(self, xi, xj):
        return np.exp((-1. * self.gamma) * np.power(np.linalg.norm(xi - xj), 2))

    def initialize_hessian_mat(self):
        dim = self.train_y.shape[0]
        gram_mat = np.zeros(shape=(dim, dim))
        for i, xi in enumerate(self.train_x):
            for j, xj in enumerate(self.train_x):
                gram_mat[i][j] = self.rbf_kernel(xi, xj)
        y = np.diag(self.train_y.ravel())
        self.hessian_mat = y.dot(gram_mat).dot(y)

    def objective_function(self):
        F = (self.lambda_star.T.dot(self.hessian_mat).dot(
            self.lambda_star)) / 2.0 + self.e.T.dot(self.lambda_star)
        return F[0][0]

    def update_gradient(self, lambda_k1, lambda_k2, working_set):
        hessian_mat_w_cols = self.hessian_mat.T[np.ix_(working_set)].T
        delta_lambda = lambda_k1[working_set] - lambda_k2[working_set]
        return hessian_mat_w_cols.dot(delta_lambda)  # delta_gradient

    def predict(self, data_x):
        threshold = 0
        for i, sample in enumerate(self.train_x):
            threshold += self.lambda_star[i] * \
                         self.train_y[i] * self.rbf_kernel(sample, data_x)
        return np.sign(threshold + self.bias)

    def acc(self, test_x, test_y):
        correctly_classified = 0
        data_len = test_y.shape[0]
        for i, label in enumerate(test_y):
            y_hat = self.predict(test_x[i])
            if y_hat * label > 0:
                correctly_classified += 1
        return correctly_classified / data_len

    def build_mat(self, working_set, not_working_set, lambda_):
        train_y = self.train_y.reshape(len(self.train_y), 1)
        hessian_mat_ww = self.hessian_mat[np.ix_(working_set, working_set)]
        e_ww = self.e[working_set]
        lambda_not_w = lambda_[not_working_set]
        hessian_mat_not_ww = self.hessian_mat[np.ix_(not_working_set,
                                                     working_set)]

        y_w, y_not_w = train_y[working_set], train_y[not_working_set]

        P = matrix(hessian_mat_ww, tc='d')
        c = matrix((lambda_not_w.T.dot(hessian_mat_not_ww) + e_ww.T).T, tc='d')
        A = matrix(y_w, (1, y_w.shape[0]), tc='d')
        b = matrix(-y_not_w.T.dot(lambda_not_w), tc='d')

        G = matrix(np.vstack((-np.eye(y_w.shape[0]),
                              np.eye(y_w.shape[0]))),
                   tc='d')
        h = matrix(np.hstack((np.zeros(y_w.shape[0]),
                              self.C * np.ones(y_w.shape[0]))),
                   tc='d')
        return P, c, G, h, A, b

    def qp_solver(self, working_set, not_working_set, lambda_):
        locals = self.build_mat(working_set, not_working_set, lambda_)
        res = solvers.qp(*locals)
        self.dual_obj = res.get('dual objective')
        return np.array(res.get('x')), res.get('iterations')

    def optimize(self, lambda_, q, print_info=True):
        train_y_ = self.train_y.reshape(len(self.train_y), 1)
        self.iterations, evaluations = 0, 0
        self.initialize_hessian_mat()
        # reinitialize
        self.e = -1. * np.ones(shape=(self.train_y.shape[0], 1))
        gradient = np.copy(self.e)
        tik = time.time()
        logging.info("Optimization process started. \n")
        while True:
            lambda_k = np.copy(lambda_)
            grad_y = -gradient / train_y_
            idx = np.arange(0, len(train_y_)).reshape((len(train_y_), 1))
            epsillon = 1e-5
            lambda_l = lambda_ <= epsillon
            lambda_u = np.logical_and(lambda_ >= (self.C - epsillon), lambda_ <= self.C)

            # lambda_l dim > train_y_ dim
            l_plus_cond, l_minus_cond = np.logical_and(lambda_l, train_y_ == 1), np.logical_and(lambda_l, train_y_ == -1)
            l_plus, l_minus = list(idx[l_plus_cond]), list(idx[l_minus_cond])

            u_plus_cond, u_minus_cond = np.logical_and(
                lambda_u, train_y_ == 1), np.logical_and(lambda_u, train_y_ == -1)
            u_plus, u_minus = list(idx[u_plus_cond]), list(idx[u_minus_cond])

            f_cond = np.logical_and(lambda_ > epsillon,
                                    lambda_ < (self.C - epsillon))
            f = list(idx[f_cond])

            R, S = sorted(l_plus + u_minus + f), sorted(l_minus + u_plus + f)

            m_lambda, M_lambda = grad_y[R].max(), grad_y[S].min()

            if m_lambda <= M_lambda:
                logging.info(
                    f"m - M = {m_lambda - M_lambda}, Optimization terminated successfully, by reaching K.K.T point.")
                break
            else:
                max_grad_y_R = (grad_y[R].ravel()).argsort()[:][::-1]
                min_grad_y_S = (grad_y[S].ravel()).argsort()[:]
                max_grad_y, min_grad_y = max_grad_y_R[0:int(
                    q / 2)], min_grad_y_S[0:int(q / 2)]
                I, J = [R[i] for i in max_grad_y], [S[j] for j in min_grad_y]
                working_set = I + J
                idx_set = list(idx.ravel())
                not_working_set = [j for j in idx_set if j not in working_set]
                lambda_w, num_eval = self.qp_solver(
                    working_set, not_working_set, lambda_k)
                lambda_[working_set] = np.copy(lambda_w)
                delta_grad = self.update_gradient(
                    lambda_, lambda_k, working_set)
                gradient += delta_grad
                evaluations += num_eval
                iterations += 1
        logging.info("Optimization done. \n")
        tok = time.time()
        self.computational_time = tok - tik
        self.lambda_star = lambda_
        support_vector_idx = lambda_.argmax()
        bias_x = self.train_x[support_vector_idx]
        bias_y = self.train_y[support_vector_idx]
        self.bias = (1 - bias_y * self.predict(bias_x)) / bias_y
        acc_train = self.acc(self.train_x, self.train_y)
        acc_test = self.acc(self.test_x, self.test_y)
        obj_val = self.objective_function()
        if print_info:
            self._log_info(acc_train, acc_test, obj_val,
                           iterations, self.computational_time)

        return (lambda_, acc_train, acc_test, obj_val,
                evaluations, iterations, self.computational_time)

    def _log_info(self, acc_train, acc_test, obj_value, iterations, comp_time):
        with open(self.logging_path, encoding='utf-8', mode='w') as f:
            print('\n--------------------\n', file=f)
        logging.info(f"C: {self.C}")
        logging.info(f"gamma: {self.gamma}")
        logging.info(f"Final val of objective function: {obj_value:.5f}")
        logging.info(f"Train acc: {acc_train * 100:.4f}%")
        logging.info(f"Test acc: {acc_test * 100:.4f}%")
        logging.info(f"Time to find KKT point: {comp_time:.4f} seconds")
        logging.info(f"Function evaluations: {iterations}")
        logging.info(f"Gradient evaluations: {iterations}")
