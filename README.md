# Sanhak-Lab

## 프로젝트 진행 동기
제조 산업 분야에서 실시간으로 많은 양의 데이터가 생성되고 있으나, 무수한 양의 데이터들은 메타 데이터로 발전되지 못하고 낭비로 이루어지고 있습니다. 데이터 자체가 모두 라벨링이 되지 않았기 때문에 어떤 정보가 유용한 지 정확하게 파악할 수 없습니다. 하지만, 무수한 양의 시계열 데이터를 일일이 수작업으로 직접 라벨링하는 것은 현실적으로 불가능하기 때문에 군집화를 사용하여 데이터 탐색과 분석, 이상치 탐지에 활용할 것입니다.
특히, 공정 과정에서 비슷한 특성을 가지는 시계열끼리 군집화가 원활히 이루어진다면 공정 과정의 시간과 자재, 노동력 등의 비용을 적절히 효율적으로 활용이 가능해질 것입니다.

** 공정 시계열 데이터 외에도, 모든 시계열 데이터 군집화 및 분석이 가능합니다.

## 프로젝트의 전체적인 알고리즘 방향성

![image](https://user-images.githubusercontent.com/56811654/117939417-fe43c600-b342-11eb-8660-e75ab3ce6667.png)

핵심 알고리즘 : 방대한 양의 시계열 데이터를 전처리하고, 이를 'RP 알고리즘'을 통해 이미지로 변환하여 저장 -> 해당 이미지를 CNN 기반 Autoencoder에 적용하여 특징을 추출하고 특징 벡터를 생성함 -> 해당 특징 벡터를 바탕으로 군집화를 진행하고 성능평가를 진행함
: 해당 알고리즘[시계열 데이터의 이미지화를 통한 특징 추출과 군집화]의 특수성을 활용해 논문 작성

** 그 외 알고리즘
- Wavelet[특징 추출 알고리즘] + 군집화
- 단순 Autoencoder
- Raw Data의 단순 시각화 그래프 저장 -> CNN-Autoencoder에 적용 -> 특징 추출 후, 이를 기반으로 하여 군집화
- TimeSeriesRESampler + KMeans 사용

** 군집화 알고리즘
- K-Means
- DBSCAN
- 계층적 군집화

** 성능 측정
- Accuracy 측정
- 실루엣 지수 측정

## 군집화 결과 적용 UI
: Backend에서 군집화한 결과를, Frontend UI에 얹어 구축함
- Frontend에서 분석할 시계열 데이터를 불러오면, 해당 데이터에 알고리즘을 적용하여 군집화하고, 군집화 결과를 시각적으로(그래프 활용) 표현

## 파이썬 라이브러리 설치
### anaconda prompt창에서 아래 라이브러리를 설치해주세요.

- pip install opencv-python
- pip install keras
- pip install numpy
- pip install pyts
- pip install pandas
- pip install jupyter-dash
- pip install pywavelets
- pip install tslearn
- pip install scikit-learn

## 시연 영상


https://user-images.githubusercontent.com/56811654/118953747-10071800-b998-11eb-97bb-f83138cc2c52.mp4


