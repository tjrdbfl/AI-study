# Assignment 1
# MNIST, CIFAR-10 데이터 셋의 종류를 구분할 수 있는 퍼셉트론 모델 설계

# MNIST : 28x28 크기의 흑백 이미지 데이터셋
# CIFAR-10 : 32x32 크기의 컬러 이미지
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,RMSprop
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
# -------------------------------------------------

# 1. 입력: MNIST와 CIFAR-10이 통합된 데이터셋 만들기
# MNIST 데이터셋 불러오기 (28x28x1)
(x_train_mnist,y_train_mnist),(x_test_mnist,y_test_mnist)=ds.mnist.load_data()

# CIFAR-10 데이터셋 불러오기 (32x32x3)
(x_train_cifar,y_train_cifar),(x_test_cifar,y_test_cifar)=ds.cifar10.load_data()

# MNIST 데이터셋 변환 (28x28x1) -> (32x32x3)

# MNIST 데이터셋 변환 (28x28x1) -> (32x32x3)
# MNIST 이미지를 3차원으로 변환: [28, 28] -> [28, 28, 1]
x_train_mnist = np.expand_dims(x_train_mnist, axis=-1)
x_test_mnist = np.expand_dims(x_test_mnist, axis=-1)

# tf.image.resize_with_pad(x,32,32) : 입력 이미지 x를 32 x 32 크기로 변환
# np.stack : 리스트 형태로 저장된 배열을 하나의 Numpy 배열로 묶어줌
# np.repeat(x_train_mnist_resized,3,axis=-1) : MNIST 이미지를 컬러 이미지처럼 보이도록 채널을 복제하는 과정
# 흑백 이미지는 한 픽셀 당 하나의 값(밝기) 만 필요해 -> 채널이 1개
# 컬러 이미지는 한 픽셀 당 RGB 3가지 색상 정보 -> 채널이 3개
# 마지막 축(axis=-1)이 채널을 나타내는 것 -> 마지막 축을 기준으로 복제 
x_train_mnist_resized=np.stack([tf.image.resize_with_pad(x,32,32).numpy() for x in x_train_mnist])
x_train_mnist_resized=np.repeat(x_train_mnist_resized,3,axis=-1)

x_test_mnist_resized=np.stack([tf.image.resize_with_pad(x,32,32).numpy() for x in x_test_mnist])
x_test_mnist_resized=np.repeat(x_test_mnist_resized,3,axis=-1)

# MNIST의 레이블 0-9, CIFAR-10의 레이블 10-19로 변경
# MNIST의 레이블: 0-9
# CIFAR-10의 레이블 : 0-9
# MNIST의 레이블 0, CIFAR-10의 레이블 1로 변경
# np.zeros(y_train_mnist.shape) : y_train_mnist와 같은 크기의 배열을 생성하되, 그 배열의 모든 값을 0으로 설정
y_train_mnist = np.zeros(y_train_mnist.shape)
y_test_mnist = np.zeros(y_test_mnist.shape)

y_train_cifar=np.ones(y_train_cifar.shape).flatten()
y_test_cifar=np.ones(y_test_cifar.shape).flatten()

# MNIST와 CIFAR-10 연결
x_train=np.concatenate([x_train_mnist_resized,x_train_cifar],axis=0)
x_test=np.concatenate([x_test_mnist_resized,x_test_cifar],axis=0)

x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

y_train = np.concatenate([y_train_mnist, y_train_cifar], axis=0)
y_test = np.concatenate([y_test_mnist, y_test_cifar], axis=0)

print(f"integrated x_train shape: {x_train.shape}")
print(f"integrated y_train shape: {y_train.shape}")

# -------------------------------------------------

# 2. 퍼셉트론 모델 설계

# 2-1. 신경망 구조 설정
n_input=32*32*3
n_hidden1=1024
n_hidden2=512
n_hidden3=512
n_hidden4=512
n_output=1

# 2-3. 하이퍼 매개변수 설정
batch_siz=256
n_epoch=50
k=5

# 2-4. 모델을 설계하는 함수
def build_model():
    model=Sequential()
    model.add(Dense(units=n_hidden1,activation='relu',input_shape=(n_input,), kernel_regularizer=l2(0.001)))  # L2 정규화 추가
    model.add(Dropout(0.6))  # 드롭아웃으로 과적합 방지
    model.add(Dense(units=n_hidden2,activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(units=n_hidden3,activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(units=n_hidden4,activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(units=n_output,activation='sigmoid'))
    return model

# -------------------------------------------------

# 3. 교차 검증 함수 (KFold)

# 데이터를 1차원 벡터로 변환 (flattening)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)


def cross_validation(opt_class, learning_rate):
    accuracy=[]
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # patience 감소
   
    for train_index, val_index in KFold(k).split(x_train_flat):
        x_train_fold,x_test_fold=x_train_flat[train_index],x_train_flat[val_index]
        y_train_fold,y_test_fold=y_train[train_index],y_train[val_index]
        
        dmlp=build_model()
        opt = opt_class(learning_rate=learning_rate)  # 옵티마이저 인스턴스 매번 생성
        dmlp.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        dmlp.fit(x_train_fold,y_train_fold,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test_fold,y_test_fold),callbacks=[early_stopping],verbose=2)
        
        accuracy.append(dmlp.evaluate(x_test_fold,y_test_fold,verbose=0)[1])
    return accuracy

# -------------------------------------------------

# 4. 옵티마이저별 성능 평가
optimizers=[SGD, Adam, Adagrad, RMSprop]
learning_rates=[0.001, 0.0001, 0.0001, 0.0001]
optimizer_names=['SGD','Adam','Adagrad','RMSprop']

# 교차 검증 실행
optimizer_accuracies = {}
for opt_class, lr, name in zip(optimizers, learning_rates, optimizer_names):
    accuracy = cross_validation(opt_class, lr)
    optimizer_accuracies[name] = accuracy
    print(f'{name} optimizer classification performance (accuracy) average : {np.mean(accuracy):.4f}')


# 5. 결과 시각화
plt.figure(figsize=(10,6))
for name,acc in optimizer_accuracies.items():
    plt.plot(acc,label=f'{name} (mean={np.mean(acc):.4f})')

plt.title('Optimizer Comparison : Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()