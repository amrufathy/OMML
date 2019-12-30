import gzip
import logging
import time
from os import getcwd
from os.path import join

import cvxopt as cvx
import numpy as np
from tqdm import tqdm


class SVMMultiClassClassifier:
    def __init__(self, C=100, gamma=0.3, threshold=1e-12, p_poly=1):
        self.C = C
        self.gamma = gamma

        self.data_path = join(getcwd(), 'project2', 'Data')
        self.process_dataset()
        self.train_len = len(self.train_x)

        self.lower_bound = np.zeros((len(self.train_x), 1))
        self.upper_bound = np.full((len(self.train_x), 1), C)

        self.threshold = threshold
        self.p_poly = p_poly

        k_poly = self.poly_kernel(self.train_x, self.train_x, self.p_poly)

        y1 = np.eye(self.train_len) * self.train_y2
        y4 = np.eye(self.train_len) * self.train_y4
        y6 = np.eye(self.train_len) * self.train_y6

        self.P1 = np.dot((np.dot(y1, k_poly)), y1)
        self.P4 = np.dot((np.dot(y4, k_poly)), y4)
        self.P6 = np.dot((np.dot(y6, k_poly)), y6)

        self.h = np.concatenate([self.lower_bound, self.upper_bound], 0)
        self.e = -1 * np.ones((self.train_len, 1))
        self.Gp, self.Gn = np.eye(self.train_len), -np.eye(self.train_len)
        self.G = np.concatenate([self.Gn, self.Gp], 0)

    def process_dataset(self):
        labels_path = join(self.data_path, 'train-labels-idx1-ubyte.gz')
        images_path = join(self.data_path, 'train-images-idx3-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images_ = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = images_.reshape(len(labels), 784)

        # only loading 3 classes
        index_label2 = np.where((labels == 2))
        x_label2 = images[index_label2][:1000, :].astype('float64')

        index_label4 = np.where((labels == 4))
        x_label4 = images[index_label4][:1000, :].astype('float64')

        index_label6 = np.where((labels == 6))
        x_label6 = images[index_label6][:1000, :].astype('float64')

        self.data_x = np.concatenate([x_label2, x_label4, x_label6], 0)

        permutation = np.random.permutation(len(self.data_x))

        train_x_idx = list(set(permutation[:int(
            np.ceil(len(self.data_x) * 0.3))]) ^ set(list(range(0, len(self.data_x)))))
        self.train_x = np.take(self.data_x, train_x_idx)

        test_x_idx = list(set(permutation[:int(np.ceil(len(self.data_x) * 0.3))]))
        self.test_x = np.take(self.data_x, test_x_idx)

        self.data_y2 = np.concatenate([np.ones(len(x_label2)),
                                       -np.ones(len(x_label4)),
                                       -np.ones(len(x_label6))], 0)

        self.train_y2 = np.take(self.data_y2, train_x_idx)
        self.test_y2 = self.data_y2[permutation[:int(
            np.ceil(len(self.data_y2) * 0.3))]]

        self.data_y4 = np.concatenate([-np.ones(len(x_label2)),
                                       np.ones(len(x_label4)),
                                       -np.ones(len(x_label6))], 0)
        self.train_y4 = np.take(self.data_y4, train_x_idx)
        self.test_y4 = self.data_y4[permutation[:int(
            np.ceil(len(self.data_y4) * 0.3))]]

        self.data_y6 = np.concatenate([-np.ones(len(x_label2)),
                                       -np.ones(len(x_label4)),
                                       np.ones(len(x_label6))], 0)
        self.train_y6 = np.take(self.data_y6, train_x_idx)
        self.test_y6 = self.data_y6[permutation[:int(
            np.ceil(len(self.data_y6) * 0.3))]]

        self.grnd_truth = np.concatenate([np.ones(len(x_label2)) * 0,
                                          np.ones(len(x_label4)) * 1,
                                          np.ones(len(x_label6)) * 2], 0)
        self.ground_truth = np.take(self.grnd_truth, train_x_idx)
        self.test_ground_truth = self.grnd_truth[permutation[:int(
            np.ceil(len(self.grnd_truth) * 0.3))]]

        del (self.data_x, self.data_y2,
             self.data_y4, self.data_y6,
             self.grnd_truth)

    def poly_kernel(self, x, y, p):
        return np.power((np.matmul(x, y.T) + 1), p)

    def optimize(self, P, e, G, constraint, Y, b=0):
        Q, e = cvx.matrix(P), cvx.matrix(e)
        G, h = cvx.matrix(G), cvx.matrix(constraint)
        A, b = cvx.matrix(np.array([Y])), cvx.matrix(np.full((1, 1), float(b)))
        cvx.solvers.options['maxiters'] = 50
        res = cvx.solvers.qp(Q, e, G, h, A, b)
        return (np.array(res['x']).flatten(), res['iterations'],
                res['primal objective'])

    def opt_b(self, alpha, train_x, train_y, polynomial):
        N = 1 if len(train_x) == 0 else len(train_x)
        k = self.poly_kernel(train_x, train_x, polynomial)
        b = (1 - np.dot(alpha * train_y, k))
        return (np.sum(b, 0) / float(N))

    def classify(self, print_info=True):
        cvx.solvers.options["show_progress"] = True
        tik = time.time()
        alpha0 = np.zeros(self.train_len)
        self.obj_fn = np.dot(np.dot(alpha0.T, self.P1),
                             alpha0) - np.dot(self.e.T, alpha0)

        # Taking class 2 data versus other class labels: 4 and 6
        opt_alpha1, iteration_2, fun_2 = self.optimize(self.P1, self.e, self.G,
                                                       self.h, self.train_y2)
        opt_b = self.opt_b(opt_alpha1, self.train_x,
                           self.train_y2, self.p_poly)
        _, margin_2 = self.predict(opt_alpha1, self.train_x,
                                   self.train_x, self.train_y2,
                                   self.p_poly, opt_b)

        # Taking class 4 data versus other class labels: 2 and 6
        opt_alpha2, iteration_4, fun_4 = self.optimize(self.P4, self.e, self.G,
                                                       self.h, self.train_y4)
        opt_b4 = self.opt_b(opt_alpha2, self.train_x,
                            self.train_y4, self.p_poly)
        _, margin_4 = self.predict(opt_alpha2, self.train_x,
                                   self.train_x, self.train_y4,
                                   self.p_poly, opt_b4)

        # Taking class 6 data versus other class labels: 2 and 4
        opt_alpha6, iteration_6, fun_6 = self.optimize(self.P6, self.e, self.G,
                                                       self.h, self.train_y6)
        opt_b6 = self.opt_b(opt_alpha6, self.train_x,
                            self.train_y6, self.p_poly)
        _, margin_6 = self.predict(opt_alpha6, self.train_x,
                                   self.train_x, self.train_y6,
                                   self.p_poly, opt_b6)

        max_margin_train = np.argmax([margin_2, margin_4, margin_6], 0)
        acc_train = self.acc(max_margin_train,
                             self.ground_truth, self.train_len)

        _, margin_test2 = self.predict(opt_alpha1, self.test_x,
                                       self.test_x, self.train_y2,
                                       self.p_poly, opt_b)
        _, margin_test4 = self.predict(opt_alpha2, self.test_x,
                                       self.test_x, self.train_y4,
                                       self.p_poly, opt_b4)
        _, margin_test6 = self.predict(opt_alpha6, self.test_x,
                                       self.test_x, self.train_y6,
                                       self.p_poly, opt_b6)

        max_distance_test = np.argmax([margin_test2,
                                       margin_test4,
                                       margin_test6], 0)
        acc_test = self.acc(max_distance_test,
                            self.test_ground_truth,
                            len(self.test_x))

        tok = time.time()
        computation_time = tok - tik

        if print_info:
            self.log_info(self.p_poly, acc_train, acc_test,
                          fun_2 + fun_4 + fun_6,
                          iteration_2 + iteration_4 + iteration_6,
                          computation_time)

    def predict(self, lambda_, x1, x2, y, p, b):
        k = self.poly_kernel(x2, x1, p)
        return np.sign(np.dot(lambda_ * y, k) + b), np.dot(lambda_ * y, k) + b

    def acc(self, y_hat, labels, num_samples):
        acc = 0
        for i in tqdm(y_hat, desc='Predicting'):
            if y_hat[i] == labels[i]:
                acc += 1
        return acc / num_samples

    def log_info(self, ploy, acc_train, acc_test, obj_value, iterations, time_):
        print('\n--------------------\n')
        logging.info(f"P: {ploy}")
        logging.info(f"Final val of objective function: {obj_value:.5f}")
        logging.info(f"Train acc: {acc_train * 100:.4f}%")
        logging.info(f"Test acc: {acc_test * 100:.4f}%")
        logging.info(f"Time to find KKT point: {time_:.4f} seconds")
        logging.info(f"Function evaluations: {iterations}")


if __name__ == "__main__":
    svm_classifier = SVMMultiClassClassifier()
    svm_classifier.classify()
