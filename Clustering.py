# Clustering Algorithm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

# Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Needed Library!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Accuracy를 가져오는 함수
# data_y: 정답 데이터
# pred_y: 예측 데이터
def getAccuracy(data_y, pred_y):
    count = 0
    bool_array = (data_y == pred_y)
    for correct in bool_array:
        if(correct):
            count += 1
    return count / pred_y.size

# 정답 레이블 만들기
## 원하는 클래스 레이블의 리스트를 넘긴다.
## ex) list = [1, 4, 5] -> 클래스 1번, 4번, 5번 에 대하여 레이블 생성
def getClassLabelFor(list, batch_size=190):
    y=np.array([])
    for i in list:
        y_=np.full((1, batch_size), i)[0]
        y=np.hstack([y, y_])

    return y

# Cluster Algorithm
def kmeans(dataset, n_clusters, n_init = 10, max_iter = 300, tol = e-4, normalization='standard'):

    cluster_data = KMeans(n_clusters=n_clusters, n_init = n_init, max_iter = max_iter, tol = tol).fit(dataset)
    return cluster_data


# DBSCAN
def dbscan(dataset, eps=0.5, min_samples=5, normalization='standard'):

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_data = dbscan.fit_predict(dataset)
    return cluster_data

# Spectral Clustering
def spectralClustering(dataset, n_clusters, n_init = 10, normalization='standard'):
    
    cluster_data = SpectralClustering(n_clusters=n_clusters, n_init=n_init).fit_predict(dataset)
    return cluster_data

# Hierarchical Clustering
def hierarchicalClustering(dataset, n_clusters, n_init = 10, linkage = 'ward', normalization='standard'):

    cluster_data = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage ).fit(dataset)
    return cluster_data
 