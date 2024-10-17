from random import randint

import numpy as np
from tqdm import trange

########### TO-DO ###########
# 1. Implement perceptron using the weights self.w
#   --> See: def forward(self,X):
# 2. Implement perceptron update
#   --> See: self.w = self.w
# 3. Implement prediction
#   --> See: def predict(self, X):
#       - X is data


class Perceptron:
    """
    A Perceptron is a type of artificial neural network used in supervised learning.
    It is used to classify input vectors into two classes, based on a linear function.
    """

    def __init__(self, lr=0.5, epochs=100):
        """
        Initializes the Perceptron object.

        Parameters:
        lr (float): Learning rate for the Perceptron. Default is 0.5.
        epochs (int): Number of iterations for training the Perceptron. Default is 100.
        """
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def forward(self, X):
        """
        Perceptron function.

        Parameters:
        X (numpy.ndarray): Input vectors.

        Returns:
        numpy.ndarray: Class labels.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        return
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X, y):
        """
        Trains the Perceptron on the given input vectors and labels.

        Parameters:
        X (numpy.ndarray): Input vectors.
        y (numpy.ndarray): Labels/target.

        Returns:
        list: Number of misclassified examples at every iteration.
        """
        # X --> Inputs.
        # y --> labels/target.
        # lr --> learning rate.
        # epochs --> Number of iterations.

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Initialize weights with zero
        self.w = np.zeros(X.shape[1])

        # Empty list to store how many examples were
        # misclassified at every iteration.
        miss_classifications = []

        # Training.
        for epoch in trange(self.epochs):
            n = randint(0, m - 1)
            # predict
            predictions = y - self.forward(X.T)
            # print(predictions)
            if (predictions == 0).all():
                print(f"No errors after {epoch} epochs. Training successful!")
            else:
                # sample one prediction at random
                prediction_for_update = self.forward(X[n, :])
                # update the weights of the perceptron at random
                # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
                self.w = self.w
                self.b = self.b
                # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            # Appending number of misclassified examples
            # at every iteration.
            miss_classifications.append(predictions.shape[0] - np.sum(predictions == 0))

        return miss_classifications

    def predict(self, X):
        """
        Predicts the class labels for the given input vectors.

        Parameters:
        X (numpy.ndarray): Input vectors.

        Returns:
        numpy.ndarray: Predicted class labels.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        return
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
