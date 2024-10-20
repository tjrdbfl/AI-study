import numpy as np
import cv2

classes=[]
f=open('coco.names.txt','r')
classes=[line.strip() for line in f.readlines()]
colors=np.random.uniform(0,255,size=(len(classes),3))

img=cv2.imread('yolo_test.jpg')
height,width,channels=img.shape
blob=cv2.dnn.blobFromImage(img,1.0/255,(488,488),(0,0,0),swapRB=True,crop=False)

yolo_model=cv2.dnn.readNet('./yolov3.weights','./yolov3.cfg')
layer_names=yolo_model.getLayerName()

out_layers=[layer_names[i[0]-1] for i in yolo_model.getUnconnectedOutLayers()]

yolo_model.setInput(blob)
output3=yolo_model.forward(out_layers)

class_ids,confidences,boxes=[],[],[]

for output in output3:
    for vec85 in output:
        scores=vec85[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
            w,h=int(vec85[2]*width),int(vec85[3]*height)
            x,y=int(centerx-w/2),int(centery-y/2)

            boxes.append([x,y,w,h])
            class_ids.append(class_id)
            confidences.append(confidence)

indexes=cv2.dnn.NMSBoxes(boxes,confidence,0.5,0.4)

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        text=str(classes[class_ids[i]]+'%.3f'%confidence)
        cv2.rectangle(img,(x,y),(x+w,y+h),colors[class_ids[i]],2)
        cv2.putText(img,text,(x,y+30),cv2.FRONT_HERSHEY_PLAIN,2,colors[class_ids[i]],2)