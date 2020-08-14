#!/usr/bin/env python
# coding: utf-8

import math
import pandas
import numpy as np
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import glob
import sys
from scipy import stats
from scipy import spatial
import pickle

def read_and_clean_data(paths_meal_data, paths_carb_data):
    try:
        mealdata_df = pandas.DataFrame()
        carbdata = pandas.DataFrame()
        for path_md, path_cd in zip(paths_meal_data, paths_carb_data):
            curr_meal_data = pandas.read_csv(path_md, header=None, sep='\n')
            curr_carb_data = pandas.read_csv(path_cd, header=None, names=['carb'], nrows = len(curr_meal_data))
            mealdata_df = mealdata_df.append(curr_meal_data , ignore_index=True)
            carbdata = carbdata.append(curr_carb_data, ignore_index=True)
        mealdata_df = mealdata_df[0].str.split(",", expand=True )
        mealdata_df = mealdata_df.drop(30, axis=1)
        mealdata_df = mealdata_df.astype('float64')
        mealdata_df = pandas.concat([mealdata_df, carbdata], axis=1)
        mealdata_df = mealdata_df.dropna()
        mealdata_df = mealdata_df.iloc[:,::-1]
        mealdata_df.index = range(len(mealdata_df))
        new_columns = ['carb', *(range(len(mealdata_df.columns)-1))]
        mealdata_df.columns = new_columns
        mealdata_df
        return mealdata_df
    except Exception as e:
        print("Error while processing test data from file.")
        exit(-1)

def get_ground_truth(carb_cgm_data):
    meal_bins = {}
    ground_indexes = np.zeros(len(carb_cgm_data))
    mealbin_1 = pandas.DataFrame(carb_cgm_data[carb_cgm_data['carb']==0].iloc[:,0:30])
    ground_indexes[mealbin_1.index] = 1
    start = 0
    stop = 20
    meal_bins[1] = mealbin_1
    for i in range(2,7):
        bin_data = pandas.DataFrame(carb_cgm_data[(carb_cgm_data['carb']>start) & (carb_cgm_data['carb']<=stop)].iloc[:,0:30])
        meal_bins[i] = bin_data
        ground_indexes[bin_data.index] = i
        meal_bins[i].index = range(len(bin_data))
        start+=20
        stop+=20
    return ground_indexes.astype(int), meal_bins

def windowed_max_mean(cgm_series, window = 5):
    max_mean = -np.inf
    for end in range(5, len(cgm_series)):
        start = end - 5
        max_mean = np.max([max_mean, np.mean(cgm_series[start:end])])
    return max_mean

def top_fft(cgm_series):
    y = np.fft.fft(cgm_series)
    y_abs = abs(y[1:len(y)//2 + 1])
    y_abs=y_abs[np.argsort(y_abs)]
    return y_abs[len(y_abs)-1:len(y_abs)-2:-1]

def area_curve(cgm_series):
    cgm_series = pandas.Series(cgm_series)
    area_under_curve = np.trapz(cgm_series)
    area_under_minline = np.trapz([min(cgm_series)]*len(cgm_series))
    act_area = area_under_curve - area_under_minline
    shr_cgm = cgm_series.shift()
    diff1_cgm = shr_cgm - cgm_series
    perimeter = abs(sum(diff1_cgm[1:]))
    return act_area


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

def get_zero_max(cgm_series):
    return abs(np.min(cgm_series) - np.max(cgm_series))

def get_cgm_from_df(carb_cgm_data):
    return carb_cgm_data.iloc[:,1:].to_numpy()

def scale_data(data):
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

def KMeans_SKL(data, ground_indexes, random_state=4, n=6, white=True):
    if white:
        data = whiten(data)
    skclx = KMeans(6, init='random', random_state = random_state, n_jobs=None).fit(data)
    cluster_map = {}
    for i in range(n):
        cluster_map[i] = stats.mode(ground_indexes[np.where(skclx.labels_==i)])[0][0]
    return skclx, cluster_map


def DBScan_SKL(data, ground_indexes, eps=0.5, metric= 'euclidean', min_samples=5, white=True):
    if white:
        data = whiten(data)
    skdbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
    cluster_map = {}
    for i in list(set(skdbscan.labels_)):
        cluster_map[i] = stats.mode(ground_indexes[np.where(skdbscan.labels_==i)])[0][0]
    return skdbscan, cluster_map

def cgm_velocity(cgm_series):
    min_i = np.argmin(cgm_series[:5])
    min_val = cgm_series[min_i]
    slope = 0
    max_i = np.argmax(cgm_series[5:20])
    max_i +=5
    max_val = cgm_series[max_i]
    slope = (max_val - min_val)/(5*(max_i-min_i))
    return slope

def calculate_features(data_x):
    zero_max = []
    fft = []
    rr_ratio = []
    max_mean = []
    cgm_vel = []
    for i in data_x:
        zero_max.append(get_zero_max(i))
        fft.append(top_fft(i))
        rr_ratio.append(area_curve(i))
        max_mean.append(windowed_max_mean(i))
        cgm_vel.append(cgm_velocity(i))
    feature_matrix = np.column_stack(( max_mean, fft, rr_ratio, zero_max, cgm_vel))
    return np.array(feature_matrix)

def create_and_store_model(data, DBScan_labels, KMeans_labels):
    trained_model = {"trained_data":data, "KMeans":KMeans_labels, "DBSCAN":DBScan_labels}
    pickle.dump(trained_model, open("cluster_model.pkl", "wb"))


if __name__ == "__main__":

    ## Change the below path if training data needs to be changed.
    paths_meal_data = glob.glob("./Data/mealData*.csv")
    paths_carb_data = glob.glob("./Data/mealAmountData*.csv")

    carb_cgm_data = read_and_clean_data(paths_meal_data,paths_carb_data)
    ground_indexes, ground_truth_bins = get_ground_truth(carb_cgm_data)

    cgm_nd_array= get_cgm_from_df(carb_cgm_data)

    features = calculate_features(cgm_nd_array)

    scaled_features,_ = scale_data(features)

    kmeans_clx, kmeans_clmap = KMeans_SKL(scaled_features, ground_indexes, random_state=4, white=False)
    labels_kmeans = np.vectorize(kmeans_clmap.__getitem__)(kmeans_clx.labels_)

    dbscan_clx, dbscan_clmap = DBScan_SKL(scaled_features, ground_indexes, eps=0.5, min_samples=5, white=False)
    labels_dbscan = np.vectorize(dbscan_clmap.__getitem__)(dbscan_clx.labels_)
    
    print("\nKMeans Training Accuracy: ", 100*(np.sum(labels_kmeans==ground_indexes)/ground_indexes.shape)[0], "%")
    print("KMeans SSE: ", kmeans_clx.inertia_)
    print("KMeans Labels (Ground Truth bins):\n", labels_kmeans)
    print("\n\n")
    print("DBSCAN Training Accuracy: ", 100*(np.sum(labels_dbscan==ground_indexes)/ground_indexes.shape)[0], "%")
    print("DBSCAN Labels (Ground Truth bins):\n", labels_dbscan)
    print("\n")
    create_and_store_model(features, labels_dbscan, labels_kmeans)