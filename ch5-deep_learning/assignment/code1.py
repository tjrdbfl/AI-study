import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

# MNIST,CIFAR-10 데이터셋 불러오기 
(x_train_mnist,y_train_mnist),(x_test_mnist,y_test_mnist)=ds.mnist.load_data()
(x_train_cifar,y_train_cifar),(x_test_cifar,y_test_cifar)=ds.cifar10.load_data()

# MNIST 데이터셋 변환 (28x28) -> (32x32x3)
x_train_mnist = np.expand_dims(x_train_mnist, axis=-1)
x_test_mnist = np.expand_dims(x_test_mnist, axis=-1)
x_train_mnist_resized=np.stack([tf.image.resize_with_pad(x,32,32).numpy() for x in x_train_mnist])
x_train_mnist_resized=np.repeat(x_train_mnist_resized,3,axis=-1)

x_test_mnist_resized=np.stack([tf.image.resize_with_pad(x,32,32).numpy() for x in x_test_mnist])
x_test_mnist_resized=np.repeat(x_test_mnist_resized,3,axis=-1)

# MNIST와 CIFAR-10을 이진 분류하기 위해 데이터셋의 레이블을 각각 0과 1로 설정
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

print(f"통합된 x_train 모양: {x_train.shape}")
print(f"통합된 y_train 모양: {y_train.shape}")

# -------------------------------------------------

# # 2. 퍼셉트론 모델 설계

# # 2-1. 신경망 구조 설정
# n_input=32*32*3
# n_hidden1=1024
# n_hidden2=512
# n_hidden3=512
# n_hidden4=512
# n_output=1

# # 2-3. 하이퍼 매개변수 설정
# batch_siz=256
# n_epoch=50
# k=5

# # 2-4. 모델을 설계하는 함수
# def build_model():
#     model=Sequential()
#     model.add(Dense(units=n_hidden1,activation='relu',input_shape=(n_input,)))
#     model.add(Dropout(0.5))  # Dropout으로 과적합 방지
#     model.add(Dense(units=n_hidden2,activation='relu'))
#     model.add(Dropout(0.5))  # Dropout으로 과적합 방지
#     model.add(Dense(units=n_hidden3,activation='relu'))
#     model.add(Dropout(0.5))  # Dropout으로 과적합 방지
#     model.add(Dense(units=n_hidden4,activation='relu'))
#     model.add(Dropout(0.5))  # Dropout으로 과적합 방지
#     model.add(Dense(units=n_output,activation='sigmoid'))
#     return model

# # 데이터를 1차원 벡터로 변환 (flattening)
# x_train_flat = x_train.reshape(x_train.shape[0], -1)
# x_test_flat = x_test.reshape(x_test.shape[0], -1)

# # -------------------------------------------------

# # 3. 교차 검증 함수 (KFold)

# def cross_validation(opt):
#     accuracy=[]
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # 조기 종료 설정
    
#     for train_index, val_index in KFold(k).split(x_train_flat):
#         x_train_fold,x_test_fold=x_train_flat[train_index],x_train_flat[val_index]
#         y_train_fold,y_test_fold=y_train[train_index],y_train[val_index]
        
#         dmlp=build_model()
#         dmlp.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#         dmlp.fit(x_train_fold,y_train_fold,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test_fold,y_test_fold),callbacks=[early_stopping],verbose=2)
        
#         accuracy.append(dmlp.evaluate(x_test_fold,y_test_fold,verbose=0)[1])
#     return accuracy

# # -------------------------------------------------

# # 4. 옵티마이저별 성능 평가
# # optimizers=[SGD(learning_rate=0.001),Adam(learning_rate=0.0001),Adagrad(learning_rate=0.0001),RMSprop(learning_rate=0.0001)]
# # optimizer_names=['SGD','Adam','Adagrad','RMSprop']
# optimizers=[Adam(learning_rate=0.0001)]
# optimizer_names=['Adam']

# # 교차 검증 실행
# optimizer_accuracies={}
# for opt,name  in zip(optimizers,optimizer_names):
#     accuracy=cross_validation(opt)
#     optimizer_accuracies[name]=accuracy
#     print(f'{name} 옵티마이저별 분류 성능(accuracy) 평균 : {np.mean(accuracy):.4f}')
    
# # -------------------------------------------------

# # 5. 결과 시각화
# plt.figure(figsize=(10,6))
# for name,acc in optimizer_accuracies.items():
#     plt.plot(acc,label=f'{name} (mean={np.mean(acc):.4f})')
    
# plt.title('Optimizer Comparison : Accuracy per Fold')
# plt.xlabel('Fold')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()