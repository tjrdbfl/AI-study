from sklearn import datasets
import matplotlib.pyplot as plt

#데이터셋 읽기
lfw=datasets.fetch_lfw_people(min_faces_per_person=70,resize=0.4)

plt.figure(figsize=(20,5))

for i in range(8): # 처음 8명 디스플레이
    plt.subplot(1,8,i+1)
    plt.imshow(lfw.images[i],cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])
    
plt.show()