# YOLOv3 를 이용한 물체 검출
import numpy as np
import cv2

# 욜로는 MS COCO 데이터셋에 대해 학습되었는데, 80개의 부류 이름을 읽어온다.
classes=[]
f=open('coco.names.txt','r') # YOLOv3 가 사용하는 클래스(물체 부류) 이름을 coco.names.txt 파일에서 읽어옴
classes=[line.strip() for line in f.readlines()] # f.readlines() : 파일의 모든 줄 읽기 , line.strip() : 각 줄에서 줄바꿈 문자(공백 문자 포함) 제거 -> 배열로 저장
colors=np.random.uniform(0,255,size=(len(classes,3))) # 0에서 255 사이의 실수 값으로 80개의 3차원 벡터(RGB 색상)를 랜덤하게 생성

# 테스트할 영상 읽고 전처리
# OpenCV의 DNN(Deep Neural Network) 모듈을 사용하여 이미지를 신경망에 입력하기 위해 전처리하는 코드
# 1.0/256 : 0~255 -> 0~1 사이의 값으로 변환
# (448,448) : 입력 이미지 크기
# (0,0,0) : R,G,B 채널에 대해 각각 0의 평균값을 사용한다
# swapRB=True : OpenCV에서 이미지를 읽을 때 기본적으로 BGR 순서로 읽기 때문에 , YOLO 모델에 맞추기 위해 BGR 순서 -> RGB 순서로 바꾸주는 옵션
# crop=False : 이미지를 잘라내지 않고 그대로 비율을 유지하면서 크기 조정
img=cv2.imread('yolo_test.jpg')
height,width,channels=img.shape
blob=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True,crop=False)

# cv2.dnn.readNet : YOLOv3 네트워크 모델을 로드
# yolo_model.getLayerName() : 모델에 포함된 모든 레이어의 이름을 가져오는 함수
yolo_model=cv2.dnn.readNet('./yolov3.weights','./yolov3.cfg')
layer_names=yolo_model.getLayerName()
# getUnconnectedOutLayers(): 네트워크의 출력 계층에 해당하는 레이어 인덱스를 반환하는 함수
# yolo_81,yolo_94, yolo_106 층을 알아냄
out_layers=[layer_names[i[0]-1] for i in yolo_model.getUnconnectedOutLayers()]

yolo_model.setInput(blob)
output3=yolo_model.forward(out_layers)

# 최고 부류 확률이 0.5를 넘는 바운딩 박스 모음
class_ids,confidences,boxes=[],[],[]

for output in output3:
    for vec85 in output: # 85차원의 벡터를 출력
        scores=vec85[5:] # 첫 5개의 값은 물체의 위치와 크기에 대한 정보(x, y, width, height)이며, 그 이후의 값은 물체의 클래스별 점수
        class_id=np.argmax(scores)
        confidence=scores[class_id]  # 예측된 클래스에 대한 신뢰도(확률) 값입니다. 이 값은 0~1 사이의 값
        if confidence>0.5: # 신뢰도가 50% 이상인 경우만 취함
            centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
            w,h=int(vec85[2]*width),int(vec85[3]*height)
            x,y=int(centerx-w/2),int(centery-h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 비최대 억제를 적용하여 주위에 비해 최대인 것만 남김
# 비최대 억제(Non-Maximum Suppression, NMS)를 적용하여 중복된 바운딩 박스를 제거
# 신뢰도가 0.5 이상인 박스만 남기고 나머지는 버립니다.
# Intersection over Union (IoU) 임계값입니다. 박스가 겹칠 때, IoU(겹친 영역의 비율)가 이 값보다 큰 경우 더 신뢰도가 낮은 박스를 제거
indexes=cv2.dnn.NMSBoxes(boxes,confidence,0.5,0.4)

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        text=str(classes[class_ids[i]])+'%.3f'%confidences[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),colors[class_ids[i]],2) # (x, y)는 사각형의 왼쪽 상단 좌표이고, (x + w, y + h)는 오른쪽 하단 좌표
        cv2.putText(img,text,(x,y+30),cv2.FRONT_HERSHEY_PLAIN,2,colors[class_ids[i]],2) # (x, y + 30): 텍스트가 표시될 좌표입니다. 바운딩 박스 위에 텍스트를 표시하기 위해 y 좌표를 약간 아래로 이동

        # cv2.FONT_HERSHEY_PLAIN: 글꼴을 지정

cv2.imshow("Object detection",img)
cv2.waitKey(0) # 키보드 입력을 대기하는 함수입니다. 0을 넣으면, 사용자가 아무 키나 누를 때까지 무한정 기다립니다.
cv2.destroyAllWindows() # 모든 OpenCV 창을 닫는 함수