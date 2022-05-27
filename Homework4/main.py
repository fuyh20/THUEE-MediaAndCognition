from two_layers import two_layer_net
import numpy as np


def main():
    W1 = np.array([1.0, 2.0, 0.3, 0.4, 2.0, 1.0]).reshape((3, 2))
    W2 = np.array([1.0, 0.5, 2.0, 2.0, 1.0, 1.0]).reshape((2, 3))
    x = np.array([0.6, 0.1]).reshape((2, 1))
    y = np.array([1.0, 0.0]).reshape((2, 1))
    NetWork = two_layer_net()
    NetWork.SetParams(W1, W2)
    NetWork.SGD(x, y, 0.1)


if __name__ == "__main__":
    main()
