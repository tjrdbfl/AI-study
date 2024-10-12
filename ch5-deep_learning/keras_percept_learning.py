# 케라스 프로그래밍 : 퍼셉트론 학습
# models 클래스 : Sequential 과 functional API 모델 제작 방식 제공
# Dense : 완전 연결 층
# SGD : SGD 옵티마이저
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 1. OR 데이터 구축
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y=[[-1],[1],[1],[1]]

# 2. 신경망 구조 설계
# input 데이터 : 2차원 벡터
# output 데이터: 1차원 벡터
n_input=2
n_output=1

# Sequential : 각 레이어가 순자척으로 입력을 받고, 다음 layer로 출력을 전달하는 구조를 가진 모델
# units: 출력으로 나오는 뉴런 개수
# activation : 활성화 함수, 입력을 -1과 1사이로 변환
# kernel_initializer : 가중치를 어떻게 초기화할지 정하기
# 가중치를 random 값으로 설정하여, 편향되지 않은 학습을 유도
# bias_initializer : bias 값을 0으로 초기화
perceptron=Sequential()
perceptron.add(Dense(units=n_output,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',bias_initializer='zeros'))

# 3. 신경망 학습
# loss='mse' : 손실함수, mse(Mean Squared Error)
# metrics=['mse'] : 모델 성능 평가 지표
# verbose=2 : 학습 중 각 epoch 마다 결과를 한 줄로 출력
perceptron.compile(loss='mse',optimizer=SGD(learning_rate=0.1), metrics=['mse'])
perceptron.fit(x,y,epochs=500,verbose=2)

# 4. 예측
res=perceptron.predict(x)
print(res)

