import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

cnn=tf.keras.models.load_model('./virtualization/my_cnn.h5')
cnn.summary()

res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은 ",res[1]*100)

# 컨볼루션층의 커널을 시각화
for layer in cnn.layers:
    if 'conv' in layer.name:
        kernel,biases=layer.get_weights()
        print(layer.name, kernel.shape)

# 층 0의 커널 정보를 저장 (모델의 첫 번째 층)
kernel,biases=cnn.layers[0].get_weights()
minv,maxv=kernel.min(),kernel.max()
kernel=(kernel-minv)/(maxv-minv) # 커널의 가중치 값들을 0과 1 사이로 정규화
n_kernel=32 # 첫 번째 컨볼루션 층에서 32개의 필터를 적용

import matplotlib.pyplot as plt

# kernel : CNN의 첫 번째 Conv2D 층에서 사용된 필터의 가중치 배열
# 4차원 텐서로, (height, width, input_channels, output_channels)
plt.figure(figsize=(20,3))
plt.suptitle("Kernels of conv2d_4")
for i in range(n_kernel):
    f=kernel[:,:,:,i] # i번째 필터의 가중치를 3차원 배열로 가져오는 것, :는 해당 차원에서 모든 요소, i번째 **필터(커널)**
    for j in range(3):
        plt.subplot(3,n_kernel,j*n_kernel+i+1) # (nrows, ncols, index) 의 서브 플롯 생성, 
        plt.imshow(f[:,:,j],cmap='gray') # 해당 채널의 2차원 이미지를 생성
        plt.xticks([]);plt.yticks([]) # 눈금 숨기기
        plt.title(str(i)+'_'+str(j))
plt.show()

