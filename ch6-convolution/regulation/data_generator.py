# 규제 1 - 데이터 증대
# Image Data Generator로 영상 데이터셋 증대
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# CIFAT-10 부류 이름
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck']

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype('float32');x_train/=255

# 앞 12개에 대해서만 데이터 증대 적용
x_train=x_train[0:12,]
y_train=y_train[0:12,]

# 앞 12 개 영상 그려주기
plt.figure(figsize=(16,2))
plt.suptitle("First 12 images in the train set")
for i in range(12):
    plt.subplot(1,12,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]);plt.yticks([])
    plt.title(class_names[int(y_train[i])])
    
# 영상 증대기 생성
# rotation_range=30.0 : 0~30 도 사이의 임의의 각도로 이미지를 회전
# width_shift_range=0.2,height_shift_range=0.2 : 이미지 너비의 20% 범위 내에서 수평, 수직으로 이동
# horizontal_flip=True : 좌우반전
# batch_siz : 한 번에 몇 개의 데이터를 증대하여 반환할지
batch_siz=6
generator=ImageDataGenerator(rotation_range=30.0,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
gen=generator.flow(x_train,y_train,batch_size=batch_siz)

# 첫 번재 증대하고 그리기
img,label=next(gen)  # next() 를 호출할 때마다 batch_size 매개변수가 지정한 수만큼 영상 생성
plt.figure(figsize=(16,3))
plt.suptitle("Generator trial 1")
for i in range(batch_siz):
    plt.subplot(1,batch_siz,i+1)
    plt.imshow(img[i])
    plt.xticks([]);plt.yticks([])
    plt.title(class_names[int(label[i])])

img,label=next(gen)
plt.figure(figsize=(16,3))
plt.suptitle("Generator trial 2")

for i in range(batch_siz):
    plt.subplot(1,batch_siz,i+1)
    plt.imshow(img[i])
    plt.xticks([]);plt.yticks([])
    plt.title(class_names[int(label[i])])

plt.show()


