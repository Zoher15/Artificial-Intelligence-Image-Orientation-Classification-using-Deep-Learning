#!/usr/bin/env python3
#
#
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy as np


#######
# K-nearest for value of K = 30
def knn_vectorized(test_data, data, k, expected_outputs):
    distances = cdist(data, test_data)
    idx = np.argpartition(distances, k, axis=0)[:k]
    nearest_dists = np.take(expected_outputs, idx)
    out = mode(nearest_dists,axis=0)[0]
    return out

#######
# Accuracy
def knn_accuracy(predicted_output, expected_output):
    total_count = correct_count = 0
    for i in range(len(predicted_output)):
        if (predicted_output[i] == expected_output[i]):
            correct_count += 1
        total_count += 1
    return float(correct_count * 100)/total_count

#######
# Read Train data
def read_train_data(t_fname):
    x = np.genfromtxt(t_fname, dtype="str")
    names = x[:,0]
    N = len(names)
    expected_output = x[:,1].astype(np.float)
    data = x[:,2:].astype(np.float)
    weights = np.random.random_sample(N)
    return names, expected_output, data, weights

def train_Knn(train_fname, model_fname):
    train_file = open(train_fname, "r")
    model_file = open(model_fname, "w")
    for each_line in train_file:
        model_file.write(each_line)
    return

def print_output(name, orient):
    op_file = open("output.txt", "w+")
    for i in range(len(name)):
        result = str(name[i]) + " " + str(orient[i]) + "\n"
        op_file.write(result)


###
# Main Program
def knn_main(parameter, t_fname, model_fname):
    if parameter == "train":
        train_Knn(t_fname, model_fname)
    else:
        # Read Training Data
        name, expected_output, data, weights = read_train_data(model_fname)
        # Read Testing Data
        test_pic_name, test_expected_op, test_data, test_weights = read_train_data(t_fname)
        k_value = 30
        predicted_output = knn_vectorized(test_data, data, k_value, expected_output)

        print ("accuracy is", knn_accuracy(predicted_output[0], test_expected_op))
        print_output(test_pic_name, predicted_output[0])
    return

