# 단일 하이퍼 매개변수의 최적화
# validation_curve (검증 곡선) 함수로 최적의 은닉 노드 개수 찾기
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,validation_curve
import numpy as np
import matplotlib.pyplot as plt
import time

#데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits()
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)

# 다층 퍼셉트론을 교차 검증으로 성능 평가 (소요 시간 측정 포함)
# n_jobs=4 : 코어 4개로 병렬 처리
# prange : 은닉층 노드(hidden layer)의 개수 범위
# validation_cuve : 범위에 따라 은닉층 노드 개수를 변화시키며 모델을 학습하고 검증
start=time.time()
mlp=MLPClassifier(learning_rate_init=0.001,batch_size=32,max_iter=300,solver='sgd')
prange=range(50,1001,50) # 50~1000까지 50씩 증가
train_score,test_score=validation_curve(mlp,x_train,y_train,param_name="hidden_layer_sizes",param_range=prange,scoring="accuracy",n_jobs=4)
end=time.time()
print("하이퍼 매개변수 최적화에 걸린 시간은",end-start,"초 입니다.")

# 교차 검증 결과의 평균과 분산 구하기
# train_score : 훈련 집합의 정확도
# test_score : 교차 검증에서 얻은 테스트 데이터의 정확도
# axis=1 : 각 은닉 노드 수에 대한 교차 검증 결과의 평균 정확도를 계산
train_mean=np.mean(train_score,axis=1)
train_std=np.std(train_score,axis=1)
test_mean=np.mean(test_score,axis=1)
test_std=np.std(test_score,axis=1)

# 성능 그래프 그리기
# plt.fill_between : 그래프 아래에 음영을 넣어 데이터의 분산 범위를 시각화
# loc="best" : 범례가 자동으로 그래프에서 가장 적절한 위치에 배치되도록
plt.plot(prange,train_mean,label="Train score",color="r")
plt.plot(prange,test_mean,label="Test score",color="b")
plt.fill_between(prange,train_mean-train_std,train_mean+train_std,alpha=0.2,color="r")
plt.fill_between(prange,test_mean-test_std,test_mean+test_std,alpha=0.2,color="b")
plt.legend(loc="best")
plt.title("Validation Curve with MLP")
plt.xlabel("Number of hidden nodes")
plt.ylabel("Accuracy")
plt.ylim(0.9,1.01)
plt.grid(axis='both')
plt.show()

# 최적의 은닉 노드 개수
best_number_nodes=prange[np.argmax(test_mean)]
print("\n최적의 은닝층의 노드 개수는 ",best_number_nodes,"입니다")

# 최적의 은닉 노드 개수로 모델링
mlp_test=MLPClassifier(hidden_layer_sizes=(best_number_nodes),learning_rate_init=0.001,batch_size=32,max_iter=300,solver='sgd')
mlp_test.fit(x_train,y_train)

# 테스트 집합으로 예측
res=mlp_test.predict(x_test)

# 혼동 행렬
conf=np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

# 정확률 계산
no_correct=0
for i in range(10):
    no_correct+=conf[i][i]
accuracy=no_correct/len(res)
print("테스트 집합에 대한 정확률은", accuracy*100,"%입니다.")
