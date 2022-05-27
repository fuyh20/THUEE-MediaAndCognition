import numpy as np
import matplotlib.pyplot as plt

x = np.array([i for i in range(1, 11)]).reshape(10, 1)
y_high = np.array([13, 17, 12, 12, 11, 12, 18, 1, 17, 20]).reshape(10, 1)
y_low = np.array([-3, -2, -2, -3, -3, -3, -1, -1, 2, 6]).reshape(10, 1)


def Regression(x: np.ndarray, y: np.ndarray, dim: int) -> np.ndarray:
    w = np.zeros((dim, 1))
    X = np.ones((x.shape[0], 1))
    for i in range(1, dim + 1):
        X = np.hstack((X, x**i))
    # print(X)
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w


def Draw(w: np.ndarray) -> None:
    x_hat = np.linspace(0, 10, 300).reshape((300, 1))
    X_hat = np.ones((300, 1))
    for i in range(1, w.shape[0]):
        X_hat = np.hstack((X_hat, x_hat**i))
    y_hat = X_hat.dot(w)
    plt.plot(x_hat, y_hat)


def main():
    for i in range(1, 6):
        plt.subplot(2, 3, i)
        plt.title(f"m={i}")
        plt.scatter(x, y_high, c="blue", s=10)
        plt.scatter(x, y_low, c="red", s=10)
        Draw(Regression(x, y_high, i))
        Draw(Regression(x, y_low, i))
    plt.show()


if __name__ == '__main__':
    main()
