import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import time
import uuid

STANDARD_SIZE = (50, 50)
SPEED_DETECT = {'0': '20 Km/h', '1': '30Km/h', '2': '50Km/h', '3': '60Km/h', '4': '70Km/h', '5': '80Km/h'}

class BoundingBox(object):
    def __init__(self, x, y, w, h, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label

#Load YOLO Algorithm
net=cv2.dnn.readNet("./yolo.weights","./yolo.cfg","darknet")
#To load all objects that have to be detected
classes=[]
with open("./classes.names","r") as f:
    read=f.readlines()
for i in range(len(read)):
    classes.append(read[i].strip("\n"))
#Defining layer names
layer_names=net.getLayerNames()
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]-1])
model = keras.models.load_model('./model_classification.h5')

def process(imgInput, isSaveTraffixSign=False):
    height,width,channels=imgInput.shape
    #Extracting features to detect objects
    blob=cv2.dnn.blobFromImage(imgInput,1/255.0,(416,416),(0,0,0), swapRB=True, crop=False)
                                                            #Inverting blue with red
                                                            #bgr->rgb
    #We need to pass the img_blob to the algorithm
    net.setInput(blob)
    outs=net.forward(output_layers)
    # #Displaying informations on the screen
    class_ids=[]
    confidences=[]
    boxes=[]
    for output in outs:
        for detection in output:
            #Detecting confidence in 3 steps
            scores=detection[5:]                #1
            class_id=np.argmax(scores)          #2
            confidence =scores[class_id]        #3
            if confidence >0.5: #Means if the object is detected
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                #Drawing a rectangle
                x=int(center_x-w/2) # top left value
                y=int(center_y-h/2) # top left value
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    #Removing Double Boxes

    boudingBoxs = []
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)
    for i in range(len(boxes)):
        if i in indexes:
            if classes[class_ids[i]] == 'speed limit':
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]  # name of the objects
                print(f"x= {x}, y= {y}, w = {w}, h= {h}, label = {label}")
                boudingBoxs.append(BoundingBox(x,y,int(1.1*w),int(1.1*h),label))

    result = None
    print(len(boudingBoxs))
    for i in range(len(boudingBoxs)):
        currentBox = boudingBoxs[i]
        speed = imgInput[currentBox.y: currentBox.y + currentBox.h, currentBox.x:currentBox.x + currentBox.w]
        if isSaveTraffixSign == True:
            cv2.imwrite(f"./{str(uuid.uuid4())}.png", speed)
        img = Image.fromarray(speed)

        img = img.resize(STANDARD_SIZE)
        img = np.array(img)
        img = np.array([img])
        img = img / 255
        tmp = model.predict_classes(img)
        if tmp != None and tmp[0] < 6:
            print(f"Result predict sign: {tmp}")    
            if str(tmp[0]) in SPEED_DETECT:
                    print(f"Speed limit: {SPEED_DETECT[str(tmp[0])]}")
    return result

def videoPredict(path='./videodemo.mp4', isSaveTraffixSign=False):
    cap = cv2.VideoCapture(path)
    result = None
    timeStart = time.time()
    timeEnd = time.time()
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            if(timeEnd - timeStart > 2):
                timeStart = time.time()
                process(frame, isSaveTraffixSign)
                # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            timeEnd = time.time()
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def imagePredict(path='./00291.jpg', isSaveTraffixSign=False):
    img=cv2.imread(path)
    result = None
    process(img, isSaveTraffixSign)

# imagePredict(path='./16.png', isSaveTraffixSign=False)
imagePredict(path='./00128.png', isSaveTraffixSign=True)