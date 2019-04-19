from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn import svm

import pandas as pd
import numpy as np
import getopt
import sys


kernels = ['rbf', 'linear', 'poly', 'sigmoid']
curr_kern = 0


def classify_data(train_data, test_data, train_label, test_label):
    clf = svm.SVC(gamma=0.001, kernel=kernels[curr_kern])
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    return accuracy_score(test_label, pred, normalize=True), \
        precision_score(test_label, pred), f1_score(test_label, pred)


def decompose_data(train_data, test_data, components):
    pca_decomposer = PCA(components)
    pca_decomposer.fit(train_data)
    return pca_decomposer.transform(train_data), \
        pca_decomposer.transform(test_data), \
        pca_decomposer.explained_variance_ratio_.cumsum()


def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)


def load_data():
    train_set = fetch_lfw_pairs(subset='train')
    test_set = fetch_lfw_pairs(subset='test')
    return train_set.data, test_set.data, train_set.target, test_set.target


def main(argv):
    components = 0
    try:
        opts, _ = getopt.getopt(
            argv, 'hc:', ['help', 'components'])
    except getopt.GetoptError:
        print('Invalid argument')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('pca_face.py -c <n_components>')
        elif opt in ('-c', '--components'):
            components = arg

    return components


if __name__ == "__main__":
    components = int(main(sys.argv[1:]))
    train_data, test_data, train_target, test_target = load_data()
    train_data, test_data = scale_data(train_data, test_data)
    train_data, test_data, var_cumsum = decompose_data(
        train_data, test_data, components)

    print("Cumulative variance with n_components = {0}: {1:.4%}".format(
        components, var_cumsum[-1]))

    acc_res, prec_res, f1_res = classify_data(train_data, test_data,
                                              train_target, test_target)

    print("SVM using {} kernel".format(kernels[curr_kern]))
    print("Accuracy: {:.0%}".format(acc_res))
    print("Precision: {:.0%}".format(prec_res))
    print("F1 Score: {:.0%}".format(f1_res))
