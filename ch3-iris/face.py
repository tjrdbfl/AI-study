from sklearn import datasets
import matplotlib.pyplot as plt

#데이터셋 읽기 (얼굴 사진으로 구성되어 있음)
#lfw : labeled faces in the wild
# min_faces_per_person=70 : 최소 70장 이상의 얼굴 사진이 있는 사람들만 데이터셋에 포함시켜라
# resize=0.4, 이미지 크기를40%로 축소해라. 이미지 크기가 원래 크기보다 작아져 메모리와 계산 비용을 줄임
lfw=datasets.fetch_lfw_people(min_faces_per_person=70,resize=0.4)

# 그래프 크기 설정
plt.figure(figsize=(20,5))

for i in range(8): # 처음 8명 디스플레이
    plt.subplot(1,8,i+1) # 함수는 여러 개의 이미지를 하나의 창에 나란히 배열
    plt.imshow(lfw.images[i],cmap=plt.cm.bone) # bone 색상 맵 : 흑백 이미지에 적합한 색상
    plt.title(lfw.target_names[lfw.target[i]])
    
plt.show()