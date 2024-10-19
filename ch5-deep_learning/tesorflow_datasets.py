# tensorflow가 제공하는 데이터셋의 텐서구조 확인하기
import tensorflow as tf
import tensorflow.keras.datasets as ds


# MNIST 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
yy_train=tf.one_hot(y_train,10,dtype=tf.int8)
print("MNIST: ",x_train.shape,y_train.shape,yy_train.shape)

# CIFAR-10 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
yy_train=tf.one_hot(y_train,10,dtype=tf.int8)
print("CIFAR-10: ",x_train.shape,y_train.shape,yy_train.shape)

# Boston Housing 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.boston_housing.load_data()
print("Boston Housing: ",x_train.shape, y_train.shape)

# Reuters 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.reuters.load_data()
print("Reuters: ",x_train.shape,y_train.shape)


# 출력결과
# MNIST:  (60000, 28, 28) (60000,) (60000, 10)
# CIFAR-10:  (50000, 32, 32, 3) (50000, 1) (50000, 1, 10)
# Boston Housing:  (404, 13) (404,)      
# Reuters:  (8982,) (8982,)

# MNIST
# 60,000개의 28x28 흑백 이미지로 구성
# 원-핫 인코딩된 라벨은 (60000,10) 형상을 가진다

# CIFAR-10
# 50,000개의 32x32 RGB 이미지로 구성
# 원-핫 인코딩된 라벨은 (50000,10) 형상을 가짐

# Boston Housing
# 404개의 13개의 특성을 가진 데이터로 구성됨
# 레이블 : 주택 가격

# Reuters
# 8982 개의 뉴스 기사로 구성된 텍스트 데이터셋
# 레이블 : 기사의 카테고리

# MNIST
# x_train : 60000개의 훈련 이미지로 구성되어 있으며,
# 각 이미지는 28 x 28 픽셀인 흑백 이미지
# y_train : 60,000개의 이미지에 대해 0~9 사이의 숫자로 이루어진 정수형 라벨
# yy_train : 라벨을 원-핫 인코딩, 각 라벨은 길이가 10인 벡터로 변환되어 있다. 


