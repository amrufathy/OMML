import time
from itertools import combinations, product

import cvxopt
from scipy.stats import mode
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold

from final_project.data_extraction import *

data_path = 'Data'
cvxopt.solvers.options['show_progress'] = False


class SVM:
    def __init__(self, kernel=rbf_kernel, C=1., gamma=1e-3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        """
        Full QP
        """
        n_samples, _ = X.shape

        # Objective
        K = self.kernel(X, gamma=self.gamma)
        P = cvxopt.matrix(y.T * K * y)
        q = cvxopt.matrix(-np.ones(n_samples))

        # Constraints
        #   0 <= alpha <= C
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        #  y.T.dot(alpha) = 0
        A = cvxopt.matrix(y.T)
        b = cvxopt.matrix(0.0)

        # solve QP problem
        tik = time.time()
        res = cvxopt.solvers.qp(P, q, G, h, A, b)
        tok = time.time()

        # get non-zero alphas => support vectors
        alphas = np.array(res['x'])
        self.__calculate_support_vectors(X, y, alphas)

        return tok - tik, res['iterations']

    def decomposition(self, X, y, max_iters=1000):
        """
        SMO using MVP algorithm
        """
        n_samples, _ = X.shape
        alphas = np.zeros((n_samples, 1))
        grad = -np.ones((n_samples, 1))

        y_lower, y_upper = (y < 0), (y > 0)

        K = self.kernel(X, gamma=self.gamma)
        Q = y.T * K * y

        tik = time.time()
        for k in range(max_iters):
            # step 1 get working set
            R, S = self.__get_index_sets(alphas, y_lower, y_upper)
            i, j, m, M = self.__get_working_set(grad, y, R, S)
            W = np.array([i, j])

            if m - M < 1e-3:
                print(f'KKT Conditions at {k} iters, (m - M) = {m - M}')
                break

            # step 2 get most feasible direction
            d = np.zeros((n_samples, 1))
            d[i], d[j] = y[i], -y[j]

            condition = (d[W] == 1).astype(np.int)
            step_size = condition * (self.C - alphas[W]) + (1 - condition) * alphas[W]
            d_max = step_size.min()

            Qww = Q[tuple(np.meshgrid(W, W))]
            t_star = -grad[W].T.dot(d[W]) / d[W].T.dot(Qww).dot(d[W])
            d_max = np.min([d_max, t_star])

            # step 3 update alpha
            alphas_prev = alphas.copy()
            alphas += (d_max * d)

            # step 4 update gradient
            delta = alphas[W] - alphas_prev[W]
            grad += Q[W, :].T.dot(delta)

        tok = time.time()
        self.__calculate_support_vectors(X, y, alphas)

        return tok - tik, k

    def __get_index_sets(self, alpha, y_lower, y_upper):
        eps = 1e-5
        alpha_lower = (alpha <= eps)
        alpha_upper = (alpha >= self.C - eps)

        l_pos = np.where(np.logical_and(alpha_lower, y_upper))[0]
        l_neg = np.where(np.logical_and(alpha_lower, y_lower))[0]
        u_pos = np.where(np.logical_and(alpha_upper, y_upper))[0]
        u_neg = np.where(np.logical_and(alpha_upper, y_lower))[0]
        mid = np.where(np.logical_and(alpha > eps, alpha < self.C - eps))[0]

        R = np.concatenate((l_pos, u_neg, mid), axis=0)
        S = np.concatenate((l_neg, u_pos, mid), axis=0)

        return np.sort(R), np.sort(S)

    @staticmethod
    def __get_working_set(grad, y, R, S):
        grad_y = (-y * grad.reshape(-1, 1)).ravel()

        r_idx = grad_y[R].argsort()[-1]  # max
        s_idx = grad_y[S].argsort()[0]  # min

        m, M = grad_y[R].max().round(3), grad_y[S].min().round(3)

        return R[r_idx], S[s_idx], m, M

    def __calculate_support_vectors(self, X, y, alphas):
        sv_idx = np.where((alphas > 1e-5).ravel())

        self.coeff = y[sv_idx] * alphas[sv_idx]
        self.support_vectors = X[sv_idx]

        y_pred = self.coeff.T.dot(self.kernel(self.support_vectors)).T
        b = (1 / y[sv_idx]) - y_pred

        self.intercept = np.mean(b)

    def predict(self, X):
        return np.sign(np.sum(self.coeff * self.kernel(self.support_vectors, X), axis=0) + self.intercept).reshape(-1, 1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, kernel=rbf_kernel, C=1., gamma=1e-3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

        self.models = dict()

    def fit(self, X, y):
        """
        One-vs-One SVMs
        """
        classes = np.unique(y)

        tik = time.time()
        for c1, c2 in combinations(classes, 2):
            idx = np.where(np.logical_or(y == c1, y == c2))[0]

            X_sub, y_sub = X[idx], y[idx]
            y_sub[y_sub == c1] = 1
            y_sub[y_sub == c2] = -1

            svm = SVM(kernel=self.kernel, C=self.C, gamma=self.gamma)
            svm.decomposition(X_sub, y_sub)

            self.models[(c1, c2)] = svm

        tok = time.time()

        return tok - tik

    def predict(self, X):
        all_preds = []
        for (c1, c2), model in self.models.items():
            y_pred = model.predict(X).ravel()
            y_pred[y_pred == 1] = y_pred[y_pred == 1] * c1
            y_pred[y_pred == -1] = y_pred[y_pred == -1] * -c2

            all_preds.append(y_pred)

        all_preds = np.array(all_preds)

        y_pred = mode(all_preds, axis=0)[0][0]
        return y_pred.reshape(-1, 1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == y)


def tuning(X, y):
    C = [0.01, 0.1, 0.5, 1, 2, 5, 10]  # penalty factor
    gamma = np.geomspace(1e-5, 10, num=7)  # spread of gaussian function

    best_score = 0
    best_params = None

    # grid search
    for c, g in product(*(C, gamma)):
        svm = SVM(C=c, gamma=g)

        # 5-fold cross validation
        avg_score = 0
        kf = KFold()
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            svm.fit(x_train, y_train)

            acc = svm.evaluate(x_test, y_test)
            avg_score = np.mean([avg_score, acc])

        print(f'Params: {c, g} => Score: {avg_score:.4f}')

        if avg_score > best_score:  # better accuracy
            best_score = avg_score
            best_params = (c, g)

    return best_params


if __name__ == '__main__':
    x_train57, y_train57, x_test57, y_test57 = load_binary_mnist(data_path)
    # tuning(x_train57, y_train57)

    c, g = 1.0, 1e-3
    svm = SVM(C=c, gamma=g)
    svm.fit(x_train57, y_train57)
    print(f'Train accuracy: {svm.evaluate(x_train57, y_train57):.4f}')
    print(f'Test accuracy: {svm.evaluate(x_test57, y_test57):.4f}')
