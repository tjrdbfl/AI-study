from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# load_digits() : 손글씨 숫자 데이터셋 불러오기
# 손글씨 숫자 데이터셋 불러오기 
# (0~9 사이의 숫자는 8x8) 크기의 흑백 이미지로 표현한 데이터
digit=datasets.load_digits()

#svm의 분류기 모델 SVC 를 생성
s=svm.SVC(gamma=0.1,C=10)
s.fit(digit.data,digit.target) # digit 데이터로 모델링


# 그래프 크기 (5x5)
plt.figure(figsize=(5,5))
# digit.image[0] : 첫 번째 손글씨 숫자 이미지 가져오기
# 이미지는 흑백으로 _r 은 반전된 흑백
# 이미지가 매끄럽게 보이지 않고 픽셀 단위로 표시됨
plt.imshow(digit.images[0],cmap=plt.cm.gray_r,interpolation='nearest')

plt.show()
# 첫 번째 숫자 이미지 데이터를 1차원 벡터로 출력
print(digit.data[0])
print("이 숫자는 ",digit.target[0],"입니다.")
