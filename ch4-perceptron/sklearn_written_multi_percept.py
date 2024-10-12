# sklearn 필기 숫자 데이터에 다층 퍼셉트론 적용
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

digit=datasets.load_digits()
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)

# MLP 분류기 모델을 학습
# batch_size 32개 : 한 번의 가중치 업데이트 동안 32개의 샘플을 사용
# solver='sgd' : 가중치 업데이트 수행 최적화 알고리즘
#sgd 는 확률적 경사 하강법 (Stochastic Gradient Descent)
mlp=MLPClassifier(hidden_layer_sizes=(100),learning_rate_init=0.001,batch_size=32,max_iter=300,solver='sgd',verbose=True)
mlp.fit(x_train,y_train)

# 테스트 집합으로 예측
res=mlp.predict(x_test)

# 혼동 행렬
conf=np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

#정확률 계산
no_correct=0  #맞춘 갯수
for i in range(10):
    no_correct+=conf[i][i]
accuracy=no_correct/len(res)
print("테스트 집합에 대한 정확률은 ", accuracy*100,"%입니다.");   
