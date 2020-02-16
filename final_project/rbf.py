from sklearn.metrics.pairwise import euclidean_distances

from final_project.data_extraction import load_binary_mnist
from final_project.utils import *


class RBF:
    def __init__(self, input_size, hidden_size=100, output_size=2):
        self.C = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (input_size + hidden_size)),
                                  size=(input_size, hidden_size))

        self.V = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (hidden_size + output_size)),
                                  size=(hidden_size, output_size))

        self._sigma = 1.

    def fit(self, X, y):
        for x_batch, y_batch in batch_generator(X, y):
            # feed forward
            distances = euclidean_distances(x_batch, self.C.T)
            vgauss = np.vectorize(self.gaussian)
            G = vgauss(np.tanh(distances))

            logits = G.dot(self.V)

            # Compute the loss and the initial gradient
            loss = softmax_cross_entropy_with_logits(logits, y_batch)
            loss_grad = grad_softmax_cross_entropy_with_logits(logits, y_batch)

            # Back propagate gradients through the network
            grad_v = np.dot(G.T, loss_grad)
            self.V -= 0.1 * grad_v

            grad_in = self.dgaussian(G) * np.dot(loss_grad, self.V.T)
            grad_weights = np.dot(x_batch.T, grad_in)
            self.C -= 0.1 * grad_weights

    def forward(self, X):
        distances = euclidean_distances(X, self.C.T)
        vgauss = np.vectorize(self.gaussian)
        G = vgauss(distances)

        return G.dot(self.V)

    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(axis=-1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == y)

    def gaussian(self, z):
        return np.exp(-np.square(z / self._sigma))

    def dgaussian(self, z):
        return (2 / np.square(self._sigma)) * self.gaussian(z) * z


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_binary_mnist('Data')
    y_train, y_test = y_train.ravel(), y_test.ravel()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    rbf = RBF(input_size=X_train.shape[1])
    for epoch in range(25):
        rbf.fit(X_train, y_train)

        print("Epoch", epoch)
        print("Train accuracy:", rbf.evaluate(X_train, y_train))
