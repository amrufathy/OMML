import numpy as np

SEED = 1848399
np.random.seed(SEED)


def softmax_cross_entropy_with_logits(logits, y_true):
    logits_pred = logits[np.arange(len(logits)), y_true.astype(int)]
    return - logits_pred + np.log(np.sum(np.exp(logits), axis=-1))


def grad_softmax_cross_entropy_with_logits(logits, y_true):
    ones_pred = np.zeros_like(logits)
    ones_pred[np.arange(len(logits)), y_true.astype(int)] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_pred + softmax) / logits.shape[0]


def binary_cross_entropy_with_logits(logits, y_true):
    y_true = np.expand_dims(y_true, -1)
    return np.sum(np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(- np.abs(logits))))


def grad_binary_cross_entropy_with_logits(logits, y_true):
    y_true = np.expand_dims(y_true, -1)
    return (1 / (1 + np.exp(- logits))) - y_true


def batch_generator(X, y, batch_size=32, shuffle=False):
    if not shuffle:
        perm = np.array(range(len(X)))
    else:
        perm = np.random.permutation(len(X))

    for start in range(0, len(X), batch_size):
        end = start + batch_size

        yield X[perm[start:end]], y[perm[start:end]]
