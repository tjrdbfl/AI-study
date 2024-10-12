from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits()
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)
# train_size 0.6은 train_test_split 함수로 데이터를 훈련집합(60%) & 테스트 집합(40%)로 나눔
# x_train, y_train : 훈련용 데이터와 레이블
# x_test, y_test : 테스트용 데이터와 레이블

# svm 의 분류 모델 SVC 를 학습
s=svm.SVC(gamma=0.001)
s.fit(x_train,y_train)

res=s.predict(x_test)

# 혼동 행렬 구하기
conf=np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1 # res[i] : 예측한 값, y_test[i] : 실제 값
print(conf)

# 정확률 측정 및 출력
no_correct=0
for i in range(10):
    no_correct+=conf[i][i]  #정확하게 에측된 값 세기
accuracy=no_correct/len(res)
print("테스트 집합에 대한 정확률은", accuracy*100,"%입니다.")