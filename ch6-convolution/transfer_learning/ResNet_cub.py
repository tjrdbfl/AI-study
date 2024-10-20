# ImageNet으로 학습된 ResNet50 을 cub 데이터셋으로 전이 학습
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
import os

# 해당 폴더에 데이터 미리 저장해두기
train_folder='CUB200/train'
test_folder='CUB200/test'

# 메모리 문제로 일부 데이터만 사용
# os.listdir(train_foler) : train_folder라는 폴더 안에 있는 파일 및 서브 폴더들의 이름을 문자열 리스트로 반환
class_reduce=0.1 # 부류 수를 줄여서 데이터양 줄이기 (속도와 메모리 효율을 위해)
no_class=int(len(os.listdir(train_folder))*class_reduce) # 부류 개수

# 훈련집합 읽어오기
# enumerate() : 각 요소와 함께 그 요소의 인덱스를 반환
x_train,y_train=[],[]
for i, class_name in enumerate(os.listdir(train_folder)): # class_name = '001.Black_footed_Albatross'
    if i < no_class:
        for fname in os.listdir(train_folder+'/'+class_name):
            img=image.load_img(train_folder+'/'+class_name+'/'+fname,target_size=(224,224)) # 이미지 파일을 읽고, 사이즈를 (224,224)로 변경해서 img 변수에 저장
            if len(img.getbands())!=3: # 색상 채널 정보를 반환 (RGB)
                print("주의 : 유효하지 않은 영상 발생", class_name,fname)
                continue
            x=image.img_to_array(img)
            x=preprocess_input(x) # preprocess_input은 전이 학습에 사용되는 사전 학습된 모델(여기서는 ResNet50)에 맞게 이미지를 전처리
            x_train.append(x)
            y_train.append(i)

# 테스트 집합 읽어오기
x_test,y_test=[],[]
for i, class_name in enumerate(os.listdir(test_folder)):
    if i < no_class:
        for fname in os.listdir(test_folder+'/'+class_name):
            img=image.load_img(test_folder+'/'+class_name+'/'+fname,target_size=(224,224))
            if len(img.getbands())!=3:
                print("주의 : 유효하지 않은 영상 발생", class_name,fname)
                continue
            x=image.img_to_array(img) # Pillow 이미지를 NumPy 배열로 변환
            x=preprocess_input(x)
            x_test.append(x)
            y_test.append(i)

x_train=np.asarray(x_train) # NumPy 배열을 리스트로 묶어 놓은 것 -> 이 리스트 자체를 하나의 NumPy 배열로 변환
y_train=np.asarray(y_train)
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)
y_train=tf.keras.utils.to_categorical(y_train,no_class)
y_test=tf.keras.utils.to_categorical(y_test,no_class)

# weights='imagenet' : ResNet50 모델을 ImageNet 데이터셋으로 미리 학습된 가중치를 불러옵
# include_top=False : 모델의 뒷부분, FC 1000 과 softmax 층을 떼내라 (특징 추출 부분만 남기기)
base_model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

# 동결 방식 : 컨볼루션 층의 가중치를 동결하여 수정이 일어나지 않게 제한
# base_model.trainable=False

cnn=Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024,activation='relu')) # 뒷 부분을 새로 부착
cnn.add(Dense(no_class,activation='softmax'))

# 미세 조정 방식의 학습 (낮은 학습률 설정. 모델의 가중치 천천히 업데이트)
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(0.00002),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=16, epochs=10,validation_data=(x_test,y_test),verbose=1)

res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은", res[1]*100)