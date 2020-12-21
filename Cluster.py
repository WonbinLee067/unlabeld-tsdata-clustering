# Clustering Algorithm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

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
## dataset 은 특징값들의 벡터 리스트(numpy)를 넘긴다.
## cluster_num : number of clusters
def kmeans(dataset, n_clusters, normalization='standard'):

    scaled_dataset = []
    if normalization == 'standard':
        scaler = StandardScaler().fit(dataset)
        scaled_dataset = scaler.transform(dataset)
    elif normalization == 'minmax':
        scaler = MinMaxScaler().fit(dataset)
        scaled_dataset = scaler.transform(dataset)
    else:
        print("정규화 진행 안함")

    cluster_data = KMeans(n_clusters=n_clusters).fit(scaled_dataset)
    return cluster_data, scaled_dataset

## epsilon : distance between nodes
## min_samples : 메인 노드가 가져야 하는 최소 노드 개수
def dbscan(dataset, eps=0.5, min_samples=5, normalization='standard'):
    if normalization == 'standard':
        scaler = StandardScaler().fit(dataset)
        scaled_dataset = scaler.transform(dataset)
    elif normalization == 'minmax':
        scaler = MinMaxScaler().fit(dataset)
        scaled_dataset = scaler.transform(dataset)
    else:
        print("정규화 진행 안함")
    
    algorithm == 'dbscan'
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_data = dbscan.fit_predict(scaled_dataset)

    return cluster_data, scaled_dataset

## 그래프로 나타내기 (2차원 경우 가능)
# cluster_result: 리스트 
# class_num : 클래스 넘버 리스트
def visualization_clusters(dataset, cluster_result, class_num):
    df=np.hstack([dataset, cluster_result.reshape(-1, 1)])
    class_list = []
    for i in range(len(class_num)):
        _class = df[df[:, 2]==class_num[i], :]
        
        class_list.append(_class)
        
    class_list = np.array(class_list)
    for i in range(len(class_num)):
        plt.scatter(class_list[i][:, 0], class_list[i][ :, 1], label="class{}".format(i), cmap='Pairs')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title("result")
    plt.show()
