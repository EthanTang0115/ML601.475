import os
import argparse
import pickle
import numpy as np
import math
from scipy.sparse import hstack

import models
from data import load_data


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    parser.add_argument("--data", type=str, required=True, help="The data file to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create (for training) or load (for testing).")
    parser.add_argument("--algorithm", type=str,
                        help="The name of the algorithm to use. (Only used for training; inferred from the model file at test time.)")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create. (Only used for testing.)")
    #parser.add_argument("--online-learning-rate", type=float, help="The learning rate for perceptron", default=1.0)
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for logistic regression", default=0.01)
    parser.add_argument("--online-training-iterations", type=int, help="The number of traning iterations for online methods.", default=5)
    parser.add_argument("--gd-iterations", type=int, help="The number of iterations of gradient descent to perform", default=20)
    parser.add_argument("--num-features-to-select", type=int, help="The number of features to use for logistic regression", default=-1)
    # change default of num_features_to_select to 10 to see the effects of applying feature selection
    # TODO: Add optional command-line arguments as necessary.

    args = parser.parse_args()

    return args


def check_args(args):
    mandatory_args = {'data', 'mode', 'model_file', 'algorithm', 'predictions_file'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception('Arguments that we provided are now renamed or missing. If you hand this in, you will get 0 points.')

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--model should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")

def select_features(X, y, num_features_to_select):
    # return 1D numpy array containing index of selected features
    X_c = X.copy()
    X_c = X_c.toarray()
    num_input_features  = X_c.shape[1]
    # select a random feature column to see if t is binary data
    num_unique = (np.unique(X_c)).shape[0]
    if num_unique == 2:
        IG = np.zeros(num_input_features)
        for i in range(num_input_features):
            IG[i] = info_gain(X_c[:,i], y)
    else:
        # binarize the input features
        X_mean = np.mean(X_c, axis=0)
        for i in range(num_input_features):
            X_c[:,i][X_c[:,i] < X_mean[i]] = 0
            X_c[:,i][X_c[:,i] >= X_mean[i]] = 1        
        IG = np.zeros(num_input_features)
        for i in range(num_input_features):
            IG[i] = info_gain(X_c[:,i], y)
    # find the index of features with large information gain
    index_array = (-IG).argsort()[:num_features_to_select]
    return index_array

def entropy(X_i):
    # X_i contains only bianry features
    # compute entropy in a certain feature
    length = X_i.shape[0]
    freq_1 = 0
    freq_0 = 0
    for i in range(length):
        if X_i[i] == 1:
            freq_1 += 1
        else:
            freq_0 += 1
    freq_1 /= length
    freq_0 /= length
    data_entropy = (-freq_1) * math.log(freq_1,2) + (-freq_0) * math.log(freq_0,2)
    return data_entropy
        
def info_gain(X_i, y):
    # X_i contains only binary features
    # compute information gain for a certain feature
    length_x = X_i.shape[0]
    freq_x_1 = 0
    freq_x_0 = 0
    freq_y_1_given_x_1 = 0
    freq_y_0_given_x_1 = 0
    freq_y_1_given_x_0 = 0
    freq_y_0_given_x_0 = 0
    for i in range(length_x):
        if X_i[i] == 1:
            freq_x_1 += 1
            if y[i] == 1:
                freq_y_1_given_x_1 += 1
            else:
                freq_y_0_given_x_1 += 1
        else:
            freq_x_0 += 1
            if y[i] == 1:
                freq_y_1_given_x_0 += 1
            else:
                freq_y_0_given_x_0 += 1
    if freq_x_1 != 0:
        freq_y_1_given_x_1 /= freq_x_1
        freq_y_0_given_x_1 /= freq_x_1
    if freq_x_0 != 0:
        freq_y_1_given_x_0 /= freq_x_0
        freq_y_0_given_x_0 /= freq_x_0
    freq_x_1 /= length_x
    freq_x_0 /= length_x
    # compute information gain
    freq_x_list = [freq_x_0, freq_x_1]
    freq_con_list = [freq_y_0_given_x_0, freq_y_1_given_x_0, freq_y_0_given_x_1, freq_y_1_given_x_1]
    info_gain = 0
    if freq_x_list[0] != 0:
        if freq_con_list[0] !=0:
            info_gain += freq_x_list[0]*freq_con_list[0]*math.log((freq_con_list[0]), 2)
        if freq_con_list[1] != 0:
            info_gain += freq_x_list[0]*freq_con_list[1]*math.log((freq_con_list[1]), 2)
    if freq_x_list[1] != 0:
        if freq_con_list[2] != 0:
            info_gain += freq_x_list[1]*freq_con_list[2]*math.log((freq_con_list[2]), 2)
        if freq_con_list[3] != 0:
            info_gain += freq_x_list[1]*freq_con_list[3]*math.log((freq_con_list[3]), 2)
    return info_gain
    
    
def main():
    args = get_args()
    check_args(args)
    
    if args.mode.lower() == "train":
        # Load the training data.
        X, y = load_data(args.data)
        
        # Create the model.
        # TODO: Add other algorithms as necessary.
        if args.algorithm.lower() == 'sumoffeatures':
            model = models.SumOfFeatures()
        elif args.algorithm.lower() == 'perceptron':
            model = models.Perceptron()
        elif args.algorithm.lower() == 'useless':
            model = models.Useless()
        elif args.algorithm.lower() == 'logisticregression':
            model = models.LogisticRegression()
        else:
            raise Exception('The model given by --model is not yet supported.')

        # Select features.
        num_orig_features = X.shape[1]
        index_array = np.empty(1)
        if args.num_features_to_select > 0:
            index_array = select_features(X, y, args.num_features_to_select)
            index_array = np.sort(index_array)
            X_selected = X[:, index_array[0]]
            for i in range(index_array.shape[0]):
                if i != 0:
                    X_selected = hstack([X_selected, X[:,index_array[i]]])
            X = X_selected
        
        # Train the model.
        if args.algorithm.lower() == 'perceptron':
            model.fit(X ,y, args.online_learning_rate, args.online_training_iterations)
        elif args.algorithm.lower() == 'logisticregression':
            model.fit(X, y, args.online_learning_rate, args.gd_iterations, num_orig_features, index_array)
        else:
            model.fit(X, y)

        # Save the model.
        try:
            with open(args.model_file, 'wb') as f:
                pickle.dump(model, f)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping model pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        X, y = load_data(args.data)
        
        # Load the model.
        try:
            with open(args.model_file, 'rb') as f:
                model = pickle.load(f)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading model pickle.")

        # Compute and save the predictions.
        y_hat = model.predict(X)
        invalid_label_mask = (y_hat != 0) & (y_hat != 1)
        if any(invalid_label_mask):
            raise Exception('All predictions must be 0 or 1, but found other predictions.')
        np.savetxt(args.predictions_file, y_hat, fmt='%d')
            
    else:
        raise Exception("Mode given by --mode is unrecognized.")


if __name__ == "__main__":
    main()
