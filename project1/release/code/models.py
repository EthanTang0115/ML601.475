import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        X = X.copy()
        self.num_input_features = X.shape[1]
        # TODO: Write code to make predictions.
        # When the number of features in a data point is even
        if self.num_input_features % 2 == 0:
            left = X[:,0:int(self.num_input_features/2)]
            right = X[:,int(self.num_input_features/2):self.num_input_features]
            left_sum = left.sum(axis=1);
            right_sum = right.sum(axis=1);
            left_sum = np.asarray(left_sum).flatten()
            right_sum = np.asarray(right_sum).flatten()
            return 1*np.greater_equal(left_sum, right_sum)
        else:
            left = X[:,0:int((self.num_input_features-1)/2)]
            right = X[:,int((self.num_input_features+1)/2):self.num_input_features]
            left_sum = left.sum(axis=1)
            right_sum = right.sum(axis=1)
            left_sum = np.asarray(left_sum).flatten()
            right_sum = np.asarray(right_sum).flatten()
            return 1*np.greater_equal(left_sum, right_sum)     


class Perceptron(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, y, online_learning_rate, online_training_iterations):
        # TODO: Write code to fit the model.
        X = X.copy()
        self.num_examples = X.shape[0]
        self.num_input_features = X.shape[1]
        # implementation of perceptron
        self.w = np.zeros(self.num_input_features)
        lr = online_learning_rate
        num_iter = online_training_iterations
        # loop through data to update parameters
        for i in range(num_iter):
            for j in range(self.num_examples):
                example = X[j, :]
                y_pred = np.sign(example.multiply(self.w).sum(axis=1))
                if y_pred == -1:
                    y_pred = 0
                else:
                    y_pred = 1 # include 0 and 1
                self.w = self.w + lr*(y[j]-y_pred)*example
        pass

    def predict(self, X):
        # TODO: Write code to make predictions.
        # check dimension of training and testing feature vectors
        X = X.copy()
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # use trained parameters to make prediction
        dot_products = X.multiply(self.w).sum(axis=1)
        dot_products = np.asarray(dot_products).flatten()
        positive_label_mask = dot_products >= 0
        negative_label_mask = dot_products < 0
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[negative_label_mask] = 0
        y_hat[positive_label_mask] = 1
        return y_hat
        pass


# TODO: Add other Models as necessary.
