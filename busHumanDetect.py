import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math


totalFare=0
pricePerKm=1.5
def distanceCalculations(i):
    return 3
    pass
def fareCalculations():
    global totalFare
    print(inBetweenDepaturePoints,inBetweenOnBoardingPoints)
    for i in inBetweenDepaturePoints:
        fare=distanceCalculations(i)*pricePerKm
        print(fare)
        totalFare+=fare
    for i in inBetweenOnBoardingPoints:
        fare=distanceCalculations(i)*pricePerKm/2
        print(fare)
        totalFare+=fare
    pass
    print(totalFare)
cap = cv2.VideoCapture('trialFootage.mp4')

model = YOLO("./Yolo-Weights/yolov8n.pt")
stationFrom='Meerut'
stationFromCoordinates={"lat":"12.12.54.4","long":"44.36.09"} # Meerut
stationToCoordinates={'lat':'54.45.56',"long":'45.45.45'}
currentCoordinates={'lat':'','long':''}
stationTo='Delhi'
reachedDestination=False

inBetweenDepaturePoints=[{}]
inBetweenOnBoardingPoints=[{}]

countPeople=0

ListPeople = []
dict = {}
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Resolution: {width}x{height}")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking



yelloLine = [270, 0, 270, 600]

RedLine = [173, 0, 173, 600]

totalCountUp = []


while True:
    print("starting loop",inBetweenOnBoardingPoints)
    

    success, img = cap.read()
    # imgRegion=cv2.bitwise_and(img,mask)

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        countPeople=0
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                countPeople+=1
            cv2.putText(img, str(countPeople), (110, 245), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    
    

    
        
    
    # print('count is ', entry_count)
    # print(dict)

    
    
    print('count is ', countPeople)
    print(dict,"dict")
    cv2.imshow("Image", img)
    cv2.waitKey(1)
   
