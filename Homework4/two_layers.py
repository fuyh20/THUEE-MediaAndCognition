from utlis import *


class two_layer_net():
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=2):
        self.params = {}
        self.params['W1'] = np.zeros((input_dim, hidden_dim))
        self.params['W2'] = np.zeros((hidden_dim, output_dim))

    def SetParams(self, W1, W2):
        self.params['W1'] = W1
        self.params['W2'] = W2

    def Loss(self, x, y):
        grads = {}

        a1, cache1 = affine_sigmoid_forward(x, self.params['W1'])
        print(f"z1:\n{cache1[1]}")
        print(f"a1:\n{a1}")
        y_hat, cache2 = affine_forward(a1, self.params['W2'])
        print(f"y_pred:\n{y_hat}")
        loss = np.sum((y_hat - y) ** 2)
        print(f"loss = {loss}")
        dy_hat = 2 * (y_hat - y)
        dout1, grads['W2'] = affine_backward(dy_hat, cache2)
        dx, grads['W1'] = affine_sigmoid_backward(dout1, cache1)

        return loss, grads

    def SGD(self, x, y, learning_rate):
        loss, grads = self.Loss(x, y)
        self.params['W1'] -= learning_rate * grads['W1']
        self.params['W2'] -= learning_rate * grads['W2']
        print(f"W1:\n{self.params['W1']}")
        print(f"W2:\n{self.params['W2']}")
