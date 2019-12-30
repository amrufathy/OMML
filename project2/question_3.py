import logging
import time

import numpy as np
from question_2 import SVMDecomposition


class SVMMVP(SVMDecomposition):
    def __init__(self, logging_path, C=2.5, gamma=0.01):
        super().__init__(logging_path, C, gamma)

    def optimize(self, lambda_, print_info=True):
        train_y_ = self.train_y.reshape(len(self.train_y), 1)
        self.initialize_hessian_mat()
        iterations = 0
        gradient = np.copy(self.e)
        tik = time.time()
        logging.info("Optimization process started.")
        while True:
            lambda_k = np.copy(lambda_)
            grad_y = -gradient / train_y_
            idx = np.arange(0, len(train_y_)).reshape((len(train_y_), 1))
            l_plus_cond, l_minus_cond = np.logical_and(
                lambda_ == 0, train_y_ == 1), np.logical_and(lambda_ == 0, train_y_ == -1)
            l_plus, l_minus = list(idx[l_plus_cond]), list(idx[l_minus_cond])

            u_plus_cond, u_minus_cond = np.logical_and(
                lambda_ == self.C, train_y_ == 1), np.logical_and(lambda_ == self.C, train_y_ == -1)
            u_plus, u_minus = list(idx[u_plus_cond]), list(idx[u_minus_cond])

            f_cond = np.logical_and(lambda_ > 0,
                                    lambda_ < self.C)
            f = list(idx[f_cond])

            R, S = sorted(l_plus + u_minus + f), sorted(l_minus + u_plus + f)

            m_lambda, M_lambda = round(
                grad_y[R].max(), 3), round(grad_y[S].min(), 3)

            if m_lambda <= M_lambda:
                logging.info(
                    f"m - M = {m_lambda - M_lambda}, Optimization terminated successfully, by reaching K.K.T point.")
                break
            else:
                i, j = R[grad_y[R].argmax()], S[grad_y[S].argmin()]
                d_ij = np.zeros((len(lambda_), 1))
                d_ij[i] = 1 / train_y_[i]
                d_ij[j] = -1 / train_y_[j]
                working_set = [i, j]

                Qw_cols = self.hessian_mat.T[np.ix_(working_set)].T
                Qww = Qw_cols[np.ix_(working_set)]
                dij_non_zero = d_ij[working_set]
                a = dij_non_zero.T.dot(Qww).dot(dij_non_zero)
                step_size = np.zeros((2, 1))
                for k, val in enumerate([i, j]):
                    if d_ij[val] == 1:
                        step_size[k] = self.C - lambda_[val]
                    elif d_ij[val] == -1:
                        step_size[k] = lambda_[val]
                    else:
                        return
                t_max = step_size.min()
                if a > 0:
                    t_star = -gradient[working_set].T.dot(dij_non_zero) / a
                    if t_star < t_max:
                        t_max = t_star
            lambda_ += (t_max * d_ij)
            delta_lambda = lambda_[working_set] - lambda_k[working_set]
            gradient += Qw_cols.dot(delta_lambda)
            iterations += 1
        logging.info("Optimization done.")
        tok = time.time()
        computational_time = tok - tik
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
                           iterations, computational_time)

        return (lambda_, acc_train, acc_test, obj_val,
                iterations, computational_time)
