#!/usr/bin/env python
# coding: utf-8

import math
import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import glob
import pickle


def unpickle_and_get(file_name):
    un_pkl = pickle.load( open( file_name, "rb" ))
    model = un_pkl["model"]
    pca = un_pkl["PCA"]
    scaler = un_pkl["scaler"]
    return model, pca, scaler

def read_test_data(file_name):
    file = open(file_name)
    lines = file.readlines()
    data_x = pandas.DataFrame()
    for line in lines:
        splitted = line.strip().split(",")
        splitted = splitted[:30]
        series = pandas.Series([np.float(x) for x in splitted])
        series_len = len(series)
        if(series_len<30):
            loop = series_len
            while(loop<30):
                series.loc[loop] = np.mean(series)
                loop+=1
        data_x = data_x.append(series, ignore_index = True)
    return data_x.to_numpy()

def windowed_max_mean(cgm_series, window = 5):
    max_mean = -np.inf
    for end in range(5, len(cgm_series)):
        start = end - 5
        max_mean = np.max([max_mean, np.mean(cgm_series[start:end])])
    return max_mean

def top_8_fft(cgm_series):
    y = np.fft.fft(cgm_series)
    y_abs = abs(y[1:len(y)//2 + 1])
    y_abs=y_abs[np.argsort(y_abs)]
    return y_abs[len(y_abs)-1:len(y_abs)-9:-1]


def roundness_ratio(cgm_series):
    cgm_series = pandas.Series(cgm_series)
    area_under_curve = np.trapz(cgm_series)
    area_under_minline = np.trapz([min(cgm_series)]*len(cgm_series))
    act_area = area_under_curve - area_under_minline
    shr_cgm = cgm_series.shift()
    diff1_cgm = shr_cgm - cgm_series
    perimeter = abs(sum(diff1_cgm[1:]))
    return float(perimeter**2/act_area)

def poly_fit(cgm_series, degree=9):
    time_series = list(range(0, len(cgm_series)*5, 5))
    coeff = np.polyfit(time_series, cgm_series, degree)
    poly = np.poly1d(coeff)
    x = time_series
    y = poly(x)
    return coeff

def binned_entropy(cgm_series, max_bins=5):
    hist, bin_edges = np.histogram(cgm_series, bins=max_bins)
    probs = hist / cgm_series.size
    probs[probs==0] = 1
    return bin_edges, [-p * np.math.log(p) for p in probs if p != 0]

def calculate_features(data_x):
    fft = []
    rr_ratio = []
    poly_coeffs =[]
    bin_entropy = []
    max_mean = []
    for i in data_x:
        fft.append(top_8_fft(i))
        rr_ratio.append(roundness_ratio(i))
        poly_coeffs.append(poly_fit(i,7))
        bin_entropy.append(binned_entropy(i,5)[0])
        max_mean.append(windowed_max_mean(i))
    fft = np.asarray(fft)
    rr_ratio = np.asarray(rr_ratio)
    poly_coeffs = np.asarray(poly_coeffs)    
    bin_entropy = np.asarray(bin_entropy)
    max_mean = np.asarray(max_mean)
    feature_matrix = np.column_stack((fft, rr_ratio, poly_coeffs, bin_entropy, max_mean))
    return feature_matrix


def transform_data(scaler, pca, data):
    features = calculate_features(data)
    scaled_features = scaler.transform(features)
    features_pca = pca.transform(scaled_features)
    return features_pca

def predict_labels(test_data, model):
    return model.predict(test_data)

def transform_and_predict(test_file_name="testdata.csv", model_file="model.pkl"):
    """
    Parameters:
    test_file_name (string) -- File name for test data with .csv extension
    model_file (string) --  File name of pickled model from train.py

    Returns:
    List: Predicted class labels of test data.
    """
    model, pca, scaler = unpickle_and_get(model_file)
    test_data = read_test_data(test_file_name)
    transformed_test_data = transform_data(scaler, pca, test_data)
    return predict_labels(transformed_test_data, model)

if __name__ == "__main__": 
    # Note: Please change the file name below to change Test file used for prediction
    test_results = transform_and_predict("testdata.csv", "model.pkl")
    print(test_results)