import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def affine_forward(x: np.ndarray, w: np.ndarray):
    out = w.dot(x)
    cache = (x, w)
    return out, cache


def sigmoid_foward(x: np.ndarray) -> np.ndarray:
    out = sigmoid(x)
    cache = x
    return out, cache


def affine_backward(dout: np.ndarray, cache):
    x, w = cache
    dx = w.T.dot(dout)
    dw = dout.dot(x.T)
    return dx, dw


def sigmoid_backward(dout: np.ndarray, cache: np.ndarray):
    dx, x = None, cache
    dx = dout * sigmoid(x) * (1 - sigmoid(x))
    return dx


def affine_sigmoid_forward(x: np.ndarray, w: np.ndarray):
    z, fc_cache = affine_forward(x, w)
    a, sigmoid_cache = sigmoid_foward(z)
    cache = (fc_cache, sigmoid_cache)
    return a, cache


def affine_sigmoid_backward(dout: np.ndarray, cache):
    fc_cache, sigmoid_cache = cache
    da = sigmoid_backward(dout, sigmoid_cache)
    dx, dw = affine_backward(da, fc_cache)
    return dx, dw
