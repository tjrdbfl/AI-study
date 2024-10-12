import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. MNIST 읽어 와서 신경망에 입력할 형태로 변환
# reshape : 2차원 구조 -> 1차원 구조 텐서로 변환
# 원래는 (28,28) 크기의 2차원 배열이었지만, 이를 784개의 요소(픽셀값)로 구성된 1차원 벡터로 변환
# astype(np.float32)/255.0 : float32 데이터형으로 변환하고 [0,255] 범위를 [0,1] 범위로 정규화
(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

# to_categorical :  각 레이블을 신경망에서 다룰 수 있는 벡터 형식으로 변환
# 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (10차원 벡터)
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

# 2. 신경망 구조 설계
# 첫 번째 층 : 입력 데이터의 형태를 받아서 10개의 뉴런으로 출력
# 두 번째 층 : 출력 뉴런 개수와 활성화 함수로 출력
n_input=784
n_hidden=1024
n_output=10

mlp=Sequential()
mlp.add(Dense(units=n_hidden,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',bias_initializer='zeros'))
mlp.add(Dense(units=n_output,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))

# 3. 신경망 학습
# batch_size : 전체 데이터셋을 학습할 때, 모델은 데이터를 한 번에 모두 처리하는 것이 아닌, 일정한 크기의 데이터 묶음인 batch 단위로 나누어 학습
# 즉, 128개의 샘플을 1번에 신경망에 입력으로 넣어 학습 시킨 후, 가중치가 업데이트 됨
mlp.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=mlp.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 4. 학습된 신경망으로 예측
# mlp.evaluate 는 2가지 값을 반환
# res[0] : 손실 함수 값
# res[1] : 평가 지표 값(정확도, metrics=['accuracy']) 
res=mlp.evaluate(x_test,y_test,verbose=0)
print("정확률은 ",res[1]*100)

# 학습 곡선 시각화

# 정확률 곡선
# history['accuracy'] : 훈련 데이터에 대한 정확도
# history['val_accuracy'] : 검증 데이터에 대한 정확도
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='upper left')
plt.grid()
plt.show()

# 손실함수 곡선
# history['loss'] : 훈련 데이터에 대한 손실 값
# history['val_loss'] : 검증 데이터에 대한 손실 값
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='upper right')
plt.grid()
plt.show()