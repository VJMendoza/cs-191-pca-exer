from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn import svm

import pandas as pd
import numpy as np


def classify_data(train_data, test_data, train_label, test_label):
    clf = svm.SVC(gamma=0.001, kernel='linear')
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    return accuracy_score(test_label, pred, normalize=True), \
        precision_score(test_label, pred), f1_score(test_label, pred)


def decompose_data(train_data, test_data):
    pca_decomposer = PCA(.95)
    pca_decomposer.fit(train_data)
    return pca_decomposer.transform(train_data), \
        pca_decomposer.transform(test_data)


def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)


def load_data():
    train_set = fetch_lfw_pairs(subset='train')
    test_set = fetch_lfw_pairs(subset='test')
    return train_set.data, test_set.data, train_set.target, test_set.target


if __name__ == "__main__":
    train_data, test_data, train_target, test_target = load_data()
    train_data, test_data = scale_data(train_data, test_data)
    train_data, test_data = decompose_data(train_data, test_data)
    acc_res, pres_res, f1_res = classify_data(train_data, test_data,
                                              train_target, test_target)
    print("Accuracy: {0:.0%} \nPrecision: {1:.0%} \nF1-score: {2:.0%}".format(
        acc_res, pres_res, f1_res))
