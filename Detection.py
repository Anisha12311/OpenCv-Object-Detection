
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_720p()
change_res(1280, 720)

objects = 'object.txt'
classNames = []
wht = 320
threshold = 0.5
newthreshold = 0.3
with open(objects,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

modelsconfigured = 'configuredect.cfg'
modelsweight = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelsconfigured,modelsweight)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findsobject(outputs,img):
    ht,wt,ct = img.shape
    classIds = []
    bbox = []
    confid = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            if confidence > threshold:
                w,h = int(det[2]*wt),int(det[3]*ht)
                x,y =  int((det[0] *wt)-w/2),int((det[1]*ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confid.append(float(confidence))
    indexes = cv.dnn.NMSBoxes(bbox,confid,threshold,newthreshold)
    print(indexes)
    for i in indexes:
        i = i[0]
        box  = bbox[i]
        x,y,h,w = box[0],box[1],box[2],box[3]
        cv.rectangle(img,(x,y),((x+h),(y+w)),(255,0,255),2)
        cv.putText(img,f'{classNames[classIds[i]].upper()}{int(confid[i]*100)}%',(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)




while True :
    success , img = cap.read()

    blob = cv.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop = False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    # print(type(outputs))
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    
    # print(outputs[0][0])
    findsobject(outputs,img)
    cv.imshow('Tracking',img)

    if cv.waitKey(1) & 0xff ==ord('d'):
        break