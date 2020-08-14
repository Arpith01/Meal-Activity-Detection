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

def read_and_clean_data(files_pointer, class_label):
    data_x = pandas.DataFrame()
    class_y = []
    if(not files_pointer):
        print("No Data Folder found. Exiting the program")
        exit(-1)
    for filename in files_pointer:
        file = open(filename)
        lines = file.readlines()
        for line in lines:
            splitted = line.strip().split(",")
            if(len(splitted)>=30):
                splitted = splitted[:30]
                series = [a.lower() for a in splitted]
                if('nan' in series):
                    continue
                else:
                    series = pandas.Series([np.float(x) for x in series])
                    data_x = data_x.append(series, ignore_index = True)
                    class_y.append(class_label)
    data_x = data_x.to_numpy()
    class_y = np.asarray(class_y)
    return data_x, class_y

def concat_data(meal_x, meal_y, no_meal_x, no_meal_y):
    total_data_x = np.row_stack((no_meal_x, meal_x))
    total_data_y = np.hstack((no_meal_y, meal_y))
    return total_data_x, total_data_y


def windowed_max_mean(cgm_series, window = 5):
    #return (float(np.argmax(cgm_series)) / len(cgm_series)) if len(cgm_series) > 0 else np.nan
    max_mean = -np.inf
    for end in range(5, len(cgm_series)):
        start = end - 5
#         print(start, end)
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

def scale_data(data):
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

def do_pca(data):
    pca = PCA(n_components=5)
    pca_data = pandas.DataFrame(pca.fit_transform(data), columns=["PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5"])
    return pca_data, pca

def get_model(total_data_x, total_data_y):
    random_forest = RandomForestClassifier(max_depth=3, bootstrap=True, random_state=0)
    random_forest.fit(total_data_x, total_data_y)
    return random_forest

def create_model(model, scaler, pca):
    model_obj = {}
    model_obj["PCA"] = pca
    model_obj["scaler"] = scaler
    model_obj["model"] = model
    return model_obj

def perform_kfold_validation(total_data_x, total_data_y):
    rf = RandomForestClassifier(max_depth=3, bootstrap=True, random_state=0)
    scores = cross_val_score(rf, total_data_x, total_data_y, cv = 10)
    print("-----KFold Cross Validation Results-----")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Scores: ",scores)

if __name__ == "__main__":
    # Note: Change folder path below inside the glob method.
    meal_fp = glob.glob(".\MealNoMealData\\mealData*.csv")
    no_meal_fp = glob.glob(".\MealNoMealData\\Nomeal*.csv")
    
    meal_x, meal_y = read_and_clean_data(meal_fp, 1)
    no_meal_x, no_meal_y = read_and_clean_data(no_meal_fp, 0)
    total_data_x, total_data_y = concat_data(meal_x, meal_y, no_meal_x, no_meal_y)
    
    #######
    # PCA only on meal Data, then transforming no meal data
    #######
    # features_meal = calculate_features(meal_x)
    # scaled_meal_x, scaler = scale_data(features_meal)
    # pca_meal_x, pca = do_pca(scaled_meal_x)
    # print(sum(pca.explained_variance_ratio_))
    # features_nomeal = calculate_features(no_meal_x)
    # scaled_no_meal_x = scaler.transform(features_nomeal)
    # pca_no_meal_x = pca.transform(scaled_no_meal_x)
    # total_data_x, total_data_y = concat_data(pca_meal_x, meal_y, pca_no_meal_x, no_meal_y)
    # perform_kfold_validation(total_data_x, total_data_y)
    # model = get_model(total_data_x, total_data_y)
    # trained_model = create_model(model, scaler, pca)
    # pickle.dump(trained_model, open("model.pkl", "wb"))
    # print("Model generated and stored at model.pkl")

    #######
    # PCA on total Data
    #######
    feature_matrix = calculate_features(total_data_x)
    scaled_features, scaler = scale_data(feature_matrix)
    pca_x, pca = do_pca(scaled_features)
    model = get_model(pca_x, total_data_y)
    trained_model = create_model(model, scaler, pca)
    perform_kfold_validation(pca_x, total_data_y)
    pickle.dump(trained_model, open("model.pkl", "wb"))
    print("Model generated and stored in model.pkl")
