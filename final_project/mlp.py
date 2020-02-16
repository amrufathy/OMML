import time

import scipy.optimize as optimize
from sklearn.model_selection import KFold

from final_project.data_extraction import load_binary_mnist
from final_project.utils import *


class MLP:
    def __init__(self, input_size, hidden_size=100, output_size=2):
        self.W = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (input_size + hidden_size)),
                                  size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)

        self.V = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (hidden_size + output_size)),
                                  size=(hidden_size, output_size))

        self.lr = 0.01
        self._rho = 1e-4
        self.hidden_size = hidden_size

        self.nfev = 0
        self.ngev = 0
        self.train_time = 0

    def forward(self, X):
        hidden_output = np.tanh(np.dot(X, self.W) + self.b1)
        return np.dot(hidden_output, self.V)

    def fit(self, X, y, max_epochs=100):
        prev_loss = np.inf

        tik = time.time()
        for i in range(max_epochs):
            epoch_loss = 0

            for x_batch, y_batch in batch_generator(X, y):
                # feed forward
                hidden_output = np.dot(x_batch, self.W) + self.b1
                hidden_activation = np.tanh(hidden_output)
                logits = np.dot(hidden_activation, self.V)

                # Compute the loss and the initial gradient
                omega = np.concatenate([self.V.flatten(), self.W.flatten(), self.b1.flatten()])
                loss = np.mean(softmax_cross_entropy_with_logits(logits, y_batch)) + \
                       0.5 * self._rho * np.square(np.linalg.norm(omega))
                loss_grad = grad_softmax_cross_entropy_with_logits(logits, y_batch) + \
                            self._rho * np.linalg.norm(omega)

                self.nfev += 1
                self.ngev += 1

                # Back propagate gradients through the network
                grad_v = np.dot(hidden_activation.T, loss_grad)
                self.V -= self.lr * grad_v

                grad_in = (1 - np.square(np.tanh(hidden_output))) * np.dot(loss_grad, self.V.T)

                grad_w = np.dot(x_batch.T, grad_in)
                grad_b = grad_in.mean(axis=0) * x_batch.shape[0]
                self.W -= self.lr * grad_w
                self.b1 -= self.lr * grad_b

                epoch_loss = np.mean([epoch_loss, loss])

            if np.abs(epoch_loss - prev_loss) < 1e-3:
                break
            prev_loss = epoch_loss

        tok = time.time()
        self.train_time = tok - tik

    def decomposition(self, X, y, max_epochs=1):
        prev_loss = np.inf

        tik = time.time()
        for _ in range(max_epochs):
            epoch_loss = 0

            for x_batch, y_batch in batch_generator(X, y):
                # feed forward
                hidden_activation = np.tanh(np.dot(x_batch, self.W) + self.b1)
                logits = np.dot(hidden_activation, self.V)

                # Compute the loss and the initial gradient
                omega = np.concatenate([self.V.flatten(), self.W.flatten(), self.b1.flatten()])
                loss = np.mean(softmax_cross_entropy_with_logits(logits, y_batch)) + \
                       0.5 * self._rho * np.square(np.linalg.norm(omega))
                loss_grad = grad_softmax_cross_entropy_with_logits(logits, y_batch) + \
                            self._rho * np.linalg.norm(omega)

                # Back propagate gradients through the network
                grad_v = np.dot(hidden_activation.T, loss_grad)
                self.V -= self.lr * grad_v

                self.W = self.W
                self.b1 = self.b1
                omega = np.concatenate([self.W.flatten(), self.b1.flatten()])
                optimal = optimize.minimize(
                    fun=self.__loss_wrapper, x0=omega, args=(x_batch, y_batch), method='L-BFGS-B',
                    tol=1e-02, options={'gtol': 1e-02, 'maxiter': 1, 'maxfun': 100, 'disp': True})
                self.W = optimal.x[:self.W.size].reshape(*self.W.shape)
                self.b1 = optimal.x[self.W.size:].reshape(*self.b1.shape)

                epoch_loss = np.mean([epoch_loss, loss])

            if np.abs(epoch_loss - prev_loss) < 1e-3:
                break
            prev_loss = epoch_loss

        tok = time.time()
        self.train_time = tok - tik

    def extreme_learning(self, X, y, max_epochs=100):
        prev_loss = np.inf

        tik = time.time()
        for i in range(max_epochs):
            epoch_loss = 0

            for x_batch, y_batch in batch_generator(X, y):
                # feed forward
                hidden_activation = np.tanh(np.dot(x_batch, self.W) + self.b1)
                logits = np.dot(hidden_activation, self.V)

                # Compute the loss and the initial gradient
                omega = np.concatenate([self.V.flatten(), self.W.flatten(), self.b1.flatten()])
                loss = np.mean(softmax_cross_entropy_with_logits(logits, y_batch)) + \
                       0.5 * self._rho * np.square(np.linalg.norm(omega))
                loss_grad = grad_softmax_cross_entropy_with_logits(logits, y_batch) + \
                            self._rho * np.linalg.norm(omega)

                self.nfev += 1
                self.ngev += 1

                # Back propagate gradients
                grad_v = np.dot(hidden_activation.T, loss_grad)
                self.V -= self.lr * grad_v

                # leave input layer weights and biases unchanged
                # -------------

                epoch_loss = np.mean([epoch_loss, loss])

            if np.abs(epoch_loss - prev_loss) < 1e-3:
                break
            prev_loss = epoch_loss

        tok = time.time()
        self.train_time = tok - tik

    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(axis=-1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == y)

    def __loss_wrapper(self, omega, X, y):
        self.W = omega[:self.W.size].reshape(*self.W.shape)
        self.b1 = omega[self.W.size:].reshape(*self.b1.shape)

        return np.sum(softmax_cross_entropy_with_logits(self.forward(X), y)) + \
               self._rho * np.linalg.norm(omega)

    def test_loss(self, X, y):
        return np.sum(softmax_cross_entropy_with_logits(self.forward(X), y))


def tuning(X, y):
    N = [25, 50, 75, 100]  # hidden units

    best_score = np.inf
    best_params = None

    for n in N:
        mlp = MLP(input_size=X.shape[1], hidden_size=n)

        # 5-fold cross validation
        avg_score = 0
        kf = KFold()
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            mlp.fit(x_train, y_train)

            err = mlp.test_loss(x_test, y_test)
            avg_score = np.mean([avg_score, err])

        if avg_score < best_score:
            best_score = avg_score
            best_params = n

    print(best_params)


if __name__ == '__main__':
    x_train57, y_train57, x_test57, y_test57 = load_binary_mnist('Data')
    y_train57, y_test57 = y_train57.ravel(), y_test57.ravel()
    y_train57[y_train57 == -1] = 0
    y_test57[y_test57 == -1] = 0

    # parameter tuning
    # tuning(x_train57, y_train57)

    mlp = MLP(input_size=x_train57.shape[1], hidden_size=100)

    mlp.fit(x_train57, y_train57)
    # mlp.decomposition(x_train57, y_train57)
    # mlp.extreme_learning(x_train57, y_train57)

    print(f'Train accuracy: {mlp.evaluate(x_train57, y_train57):.4f}')
    print(f'Test accuracy: {mlp.evaluate(x_test57, y_test57):.4f}')
    print(f'Train time: {mlp.train_time:.4f} s')
