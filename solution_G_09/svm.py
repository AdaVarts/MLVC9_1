import cvxopt
import numpy as np

########### TO-DO ###########
# 1. Implement linear kernel
#   --> See: def linear_kernel(x1, x2):
# 2. Implement rbf kernel
#   --> See: def rbf_kernel(x1, x2):
# 3. Implement fit
#   --> See: def fit(self, X, y):
#   --> Add matrix Q, p, G, h, A, b and save the solution
# 4. Implement predict
#   --> See: def predict(self, X):


class SVM:
    """Implements the support vector machine"""

    def __init__(self, kernel="linear", sigma=0.25):
        """Initialize perceptron."""
        self.__alphas = None
        self.__targets = None
        self.__training_X = None
        self.__bias = None
        if kernel == "linear":
            self.__kernel = SVM.linear_kernel
        elif kernel == "rbf":
            self.__kernel = SVM.rbf_kernel
            self.__sigma = sigma
        else:
            raise ValueError("Invalid kernel")

    @staticmethod
    def linear_kernel(x1, x2):
        """
        Computes the linear kernel between two sets of vectors.

        Args:
            x1 (numpy.ndarray): A matrix of shape (n_samples_1, n_features) representing the first set of vectors.
            x2 (numpy.ndarray): A matrix of shape (n_samples_2, n_features) representing the second set of vectors.

        Returns:
            numpy.ndarray: A matrix of shape (n_samples_1, n_samples_2) representing the linear kernel between x1 and x2.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        #print(x1.shape)
        #print(x2.shape)
        return np.dot(x1,x2.T)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    @staticmethod
    def rbf_kernel(x1, x2, sigma):
        """
        Computes the radial basis function (RBF) kernel between two sets of vectors.

        Args:
            x1: A matrix of shape (n_samples_1, n_features) representing the first set of vectors.
            x2: A matrix of shape (n_samples_2, n_features) representing the second set of vectors.

        Returns:
            A matrix of shape (n_samples_1, n_samples_2) representing the RBF kernel between x1 and x2.
        """

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        result = np.zeros((x1.shape[0], x2.shape[0]))
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                r = np.linalg.norm(x1[i] - x2[j]) # Euclidian distance
                result[i, j] = np.exp( - ( r / sigma )**2 ) # Gaußian, https://en.wikipedia.org/wiki/Radial_basis_function
        return result
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X, y):
        """Training function.

        Args:
            X (numpy.ndarray): Inputs.
            y (numpy.ndarray): labels/target.

        Returns:
            None
        """
        # n_observations -> number of training examples
        # m_features -> number of features
        n_observations, m_features = X.shape
        self.__norm = max(np.linalg.norm(X, axis=1))
        X = X / self.__norm
        y = y.reshape((1, n_observations))

        # quadprog and cvx all want 64 bits
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        print("Computing kernel matrix...")
        if self.__kernel == SVM.linear_kernel:
            K = self.__kernel(X, X)
        elif self.__kernel == SVM.rbf_kernel:
            K = self.__kernel(X, X, self.__sigma)
        print("Done.")

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # SEE: https://cvxopt.org/examples/tutorial/qp.html and https://cvxopt.org/userguide/coneprog.html#quadratic-programming and http://www.seas.ucla.edu/~vandenbe/publications/mlbook.pdf
        
        # Hessian matrix in the quadratic form
        Q = cvxopt.matrix(np.outer(y,y) * K) 

        # matrix in dual form in SVM p= -1 vector
        p = cvxopt.matrix(-np.ones((n_observations, 1))) 
        # Inequality Constraints: Gx ≤ h
        # stacked matricis shape -I and I (identity)
        G = cvxopt.matrix(np.concatenate((-np.identity(n_observations), np.identity(n_observations)), axis=0)) 
        h = cvxopt.matrix(np.concatenate((np.zeros(n_observations), np.ones(n_observations) * 1), axis=0)) 

        # Equality Constraints: Ax = b
        A = cvxopt.matrix(y, (1, n_observations)) 
        b = cvxopt.matrix(0.0) 
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Save the solution
        alphas = np.array(solution['x']).flatten()

        # Select the support vectors (where alphas > threshold)
        support_vector_indices = alphas > 1e-5
        self.__alphas = alphas[support_vector_indices]
        self.__training_X = X[support_vector_indices] # inputs at support vector indices (training)
        y = y[:, support_vector_indices].flatten()
        # Calculate bias with only average of support vector: y_i ​ − SUM(j = 1, i)(a_j * ​y_j * ​K(x_j​,x_i​))
        self.__bias = np.mean(y - np.sum(self.__alphas[:, np.newaxis] * y[:, np.newaxis] * K[support_vector_indices][:, support_vector_indices], axis=1))
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        self.__targets = y
        #self.__training_X = X

    def predict(self, X):
        """Prediction function.

        Args:
            X (numpy.ndarray): Inputs.

        Returns:
            Class label of X
        """

        X = X / self.__norm

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        if self.__kernel == SVM.linear_kernel:
            # kernel matrix between test and training set
            K = self.__kernel(X, self.__training_X)
            predictions = np.dot(K, self.__alphas * self.__targets) + self.__bias
            return np.sign(predictions)
        elif self.__kernel == SVM.rbf_kernel:
            K = self.__kernel(X, self.__training_X, self.__sigma)
            #print(self.__alphas * self.__targets.shape)
            predictions = np.dot(K, self.__alphas * self.__targets) + self.__bias
            return np.sign(predictions)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
