import numpy as np

from helpers import batch_iter


def _mean_squared_error(y, tx, w):
    residual = y - tx @ w
    return 0.5 * np.mean(residual**2)


def _mean_squared_error_gradient(y, tx, w):
    return -(tx.T @ (y - tx @ w)) / len(y)


def _sigmoid(tx, w):
    scores = tx @ w
    scores = np.clip(scores, -709, 709)
    return 1.0 / (1.0 + np.exp(-scores))


def _logistic_loss(y, tx, w):
    probabilities = np.clip(_sigmoid(tx, w), 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))


def _logistic_gradient(y, tx, w):
    return tx.T @ (_sigmoid(tx, w) - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform linear regression with batch gradient descent."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * _mean_squared_error_gradient(y, tx, w)
    return w, _mean_squared_error(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform linear regression with stochastic gradient descent."""
    w = np.array(initial_w, dtype=float, copy=True)
    for minibatch_y, minibatch_tx in batch_iter(
        y, tx, batch_size=1, num_batches=max_iters
    ):
        w -= gamma * _mean_squared_error_gradient(minibatch_y, minibatch_tx, w)
    return w, _mean_squared_error(y, tx, w)


def least_squares(y, tx):
    """Compute linear regression with the normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mean_squared_error(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Compute ridge regression with the regularized normal equations."""
    regularization = 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w = np.linalg.solve(tx.T @ tx + regularization, tx.T @ y)
    return w, _mean_squared_error(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform binary logistic regression with gradient descent."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * _logistic_gradient(y, tx, w)
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Perform L2-regularized binary logistic regression."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = _logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)
