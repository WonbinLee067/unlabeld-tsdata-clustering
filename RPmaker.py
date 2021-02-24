import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pyts.image import RecurrencePlots
from pyts.image import RecurrencePlot
import os

def readFile(filepath):
    
    return finalData


def makeRPImgFiles(dirPath, dicDatas, width=28, height=28, dpi=96, ):
    # Recurrence plot transformation
    rp = RecurrencePlots(dimension=1,epsilon='percentage_points', percentage=30)
    
    
    plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.axis('off')
    for key, values in dicDatas.items():
        # 디렉토리 생성
        dirname = key.split("_")
        path = '%s%s%s/'%(dirPath, dirname[2], dirname[3])
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            print ('Error: Creating directory. ' +  path)
        
        # 파일 존재 여부 확인 
        ## 파일 존재하면 넘어감 
        filename = '%s%s.png' %(path, key)
        if os.path.exists(filename):
            continue
        
        # RP 변환 및 이미지 파일 생성 저장
        values_rp = rp.fit_transform(values)
        plt.imshow(values_rp[0], cmap='binary', origin='lower')

        plt.savefig(filename, dpi=dpi,  bbox_inches='tight', pad_inches=0.0)

# input : Time Series Dataset 
# output : X Dataset converted into RP image data
def toRPdata(tsdatas, dimension=1, time_delay=1, threshold=None, percentage=10, flatten=False):
    X = []
    rp = RecurrencePlot(dimension= dimension,
                        time_delay= time_delay,
                        threshold= threshold,
                        percentage= percentage,
                        flatten= flatten)
    for data in tsdatas:
        data_rp = rp.fit_transform(data)
        X.append(data_rp[0])
    X = np.array(X)
    return X
    
'''
How to Use
----------------
import RPmaker as rpm

# readFile 함수 호출 (인자: 파일 경로)
## 원하는 파라미터 데이터에 대하여 파일을 읽어온다.
AXISX_data = rpm.readFile("resources/AXISX_resample.csv")
CLAMP_data = rpm.readFile("resources/CLAMP_resample.csv")
-> 결과 : 딕셔너리 키: 데이터 정보 및 이름, 값: 시계열 데이터
AXISX_data CLAMP_data 합치기?

# makeRPImgFiles 함수 호출 (인자: 저장할 폴더 경로, RP 변경할 데이터, 이미지 너비, 이미지 높이, 사용자화 dpi(default:96) )

## 파라미터 지정
## 너비와 높이는 원하는 각 너비와 높이에 4/3 곱한 값을 넣는다.
width = 37.3 -> 28 * 4/3
height = 37.3 -> 28 * 4/3
my_dpi = 96 -> 지정 필요 x
PATH = "resources/imgs28x28/" -> 저장할 경로

## 위 readFile로 불러온 데이터 삽입해야 함
rpm.makeRPImgFiles(PATH, AXISX_data, width, height, my_dpi)
rpm.makeRPImgFiles(PATH, CLAMP_data, width, height, my_dpi)
-> 결과 : 저장경로에 지정한 RP 이미지가 저장됨
'''