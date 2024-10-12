from sklearn import datasets
from sklearn import svm
import plotly.express as px

#iris 데이터셋 읽기
d=datasets.load_iris() 
print(d.DESCR)  #내용물 출력
for i in range(0,len(d.data)):  # iris 데이터 셋 읽기
    print(i+1,d.data[i],d.target[i])

#SVM(Support Vector Machine) : 기계 학습 모델    
s=svm.SVC(gamma=0.1,C=10) # SVC : svm의 분류 모델 SVC 객체 생성
s.fit(d.data,d.target) # iris 데이터로 학습시키기

new_d=[[6.4,3.2,6.0,2.5],[7.1,3.1,4.7,1.35]]
res=s.predict(new_d)
print("새로운 2개 샘플의 분류는", res)

# iris 데이터의 분포를 특징 공간에 그리기
# (차원을 하나 제외하여 3차원 공간에 데이터 분포를 그림)
df=px.data.iris()
fig=px.scatter_3d(df,x='sepal_length',y='sepal_width',z='petal_width',color='species')
fig.show(renderer="browser")
