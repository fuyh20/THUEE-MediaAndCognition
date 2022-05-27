import numpy as np


class LinearSVM(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate, reg, num_iters,
              batch_size=100, verbose=True):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            random_idxs = np.random.choice(num_train, batch_size)
            X_batch = X[random_idxs]
            y_batch = y[random_idxs]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[1])
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Question 1: Implement the regulated hinge loss function and its gradient. 

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss: a single float
        - gradient dW: an array of the same shape as W
        """

        #############################################################################
        #                             BEGIN OF YOUR CODE                            #
        #############################################################################

        loss = 0.0
        dW = np.zeros(self.W.shape)

        num_classes = self.W.shape[1]
        num_train = X_batch.shape[0]

        for i in range(num_train):
            scores = X_batch[i].dot(self.W)
            correct_class_score = scores[y_batch[i]]
            for j in range(num_classes):
                if j == y_batch[i]:
                    continue
                margin = scores[j] - correct_class_score + 1
                if margin > 0:
                    loss += margin
                    dW[:, y_batch[i]] += -X_batch[i]
                    dW[:, j] += X_batch[i]

        loss += reg * np.sum(self.W * self.W) / 2
        dW += reg * self.W

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return loss, dW
