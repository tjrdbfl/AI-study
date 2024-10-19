import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

cnn=tf.keras.models.load_model('./virtualization/my_cnn.h5')

for layer in cnn.layers:
    if 'conv' in layer.name:
        print(layer.name, layer.output.shape) # 특징 맵의 텐서 모양 출력 

from tensorflow.keras.models import Model

# 층 0의 특징맵 시각화
partial_model=Model(inputs=cnn.inputs, outputs=cnn.layers[0].output)
#partial_model.summary()

feature_map=partial_model.predict(x_test)
fm=feature_map[1]

plt.imshow(x_test[1])

plt.figure(figsize=(20,3))
plt.suptitle("Feature maps of conv2d_4")
for i in range(32):
    plt.subplot(2,16,i+1) # 2행 16열로 나누어진 플롯을 생성하고, 각 특징 맵을 그리기 위해 해당하는 서브플롯의 순서를 지정
    plt.imshow(fm[:,:,i],cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title("map"+str(i))

plt.show()
