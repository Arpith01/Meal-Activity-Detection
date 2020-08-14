import math
import pandas
import numpy as np
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

def scale_data(data):
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

def unpickle_and_get(file_name="cluster_model.pkl"):
    un_pkl = pickle.load( open( file_name, "rb" ))
    data = un_pkl["trained_data"]
    kmeans_labels = un_pkl["KMeans"]
    dbscan_labels = un_pkl["DBSCAN"]
    return un_pkl

def get_zero_max(cgm_series):
    return abs(np.min(cgm_series) - np.max(cgm_series))

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

def get_cluster_index_test_data(test_data, trained_model):
    trained_data = trained_model["trained_data"]
    labels_kmeans = trained_model["KMeans"]
    labels_DBSCAN = trained_model["DBSCAN"]
    trained_data, scaler= scale_data(trained_data)
    knn_classifier_Kmeans = KNeighborsClassifier(n_neighbors=3)
    knn_classifier_Kmeans.fit(X=trained_data, y=labels_kmeans)
    knn_classifier_DBSCAN = KNeighborsClassifier(n_neighbors=3)
    knn_classifier_DBSCAN.fit(X=trained_data, y=labels_DBSCAN)
    test_features = calculate_features(test_data)
    test_data = scaler.transform(test_features)
    return {"DBSCAN":knn_classifier_DBSCAN.predict(test_data), "KMeans":knn_classifier_Kmeans.predict(test_data)}

def read_test_data(file_name="test.csv"):
    try:
        file = open(file_name)
        lines = file.readlines()
        if(len(lines) == 0):
            raise Exception("Input File Empty!")
        data_x = pandas.DataFrame()
        for line in lines:
            splitted = line.strip().split(",")
            splitted = splitted[:30]
            # print(splitted)
            x = []
            for i in splitted:
                try:
                    x.append(np.float(i))
                except:
                    x.append(0.0)
            # series = pandas.Series([np.float(x) for x in splitted])
            series = pandas.Series(x)
            series_len = len(series)
            if(series_len<30):
                loop = series_len
                while(loop<30):
                    series.loc[loop] = np.mean(series)
                    loop+=1
            data_x = data_x.append(series, ignore_index = True)
        data_x = data_x.iloc[:,::-1]
        return data_x.to_numpy()
    except Exception as e:
        print("Error while reading test data from file.")
        print(e)
        exit(-1)

def store_output_to_csv(output, file_name = "output.csv"):
    out_df = pandas.DataFrame(output)
    print(out_df)
    out_df.to_csv(file_name, index=False, header=False)

if __name__ == "__main__":
    test_csv_file = "proj3_test.csv"
    output_file_name = "output.csv"
    if(len(sys.argv)==2):
        test_csv_file = sys.argv[1]
    if(len(sys.argv) == 3):
        output_file_name = "output.csv"

    trained_model = unpickle_and_get("cluster_model.pkl")
    test_data = read_test_data(test_csv_file)
    output = get_cluster_index_test_data(test_data, trained_model)
    store_output_to_csv(output, output_file_name)