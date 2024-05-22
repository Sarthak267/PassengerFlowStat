import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import datetime
import geocoder

#i want to take co ordinates in lat long of this device
# def get_coordinates():
#     g = geocoder.ip('me')
#     if g.latlng:
#         return {'lat': str(g.latlng[0]), 'long': str(g.latlng[1])}
#     else:
#         return None

# currentCoordinates = get_coordinates()
# if currentCoordinates:
#     print(currentCoordinates)
# else:
#      print("Unable to get device coordinates.")

#calculate distance b/w two coordinates
def distance_calculations(stationFromCoordinates, stationToCoordinates):
    stationFromCoordinates = {'lat': ' 28.98', 'long': '77.7064'}
    stationToCoordinates = {'lat': '28.66', 'long': '77.22'}
    lat1 = float(stationFromCoordinates['lat'])
    lon1 = float(stationFromCoordinates['long'])
    lat2 = float(stationToCoordinates['lat'])
    lon2 = float(stationToCoordinates['long'])
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
print(distance_calculations({'lat': ' 28.98', 'long': '77.7'}, {'lat': '28.66', 'long': '77.22'}))
totalFare=0
pricePerKm=1.5


def distanceCalculations(inBetweenDepaturePoints,inBetweenOnBoardingPoints):
    print(inBetweenDepaturePoints,inBetweenOnBoardingPoints ,'inbetween points')
    
    return 3
#calculating fare
def fareCalculations():
    global totalFare
    print(dict)
    index=0
    for i in dict.values():
        print(len(i))
        inBetweenDepaturePoints=[]
        inBetweenOnBoardingPoints=[]
        if(len(i)>3):
            print(True)
            #index of the dict
            print(index,"index")
            print(i[0],i[1],i[2],"values of the dict","key of the dict")
        
            if i[0]==True and i[1]==False:
                inBetweenDepaturePoints.append(i[2])
            if i[0]==False and i[1]==True:
                inBetweenOnBoardingPoints.append(i[2])
            fare=distanceCalculations(inBetweenDepaturePoints,inBetweenOnBoardingPoints)*pricePerKm
            print(fare,'fare')
            totalFare+=fare
        index+=1
    
    print(totalFare)
cap = cv2.VideoCapture('TrialFootage.mp4')

model = YOLO("./Yolo-Weights/yolov8n.pt")
stationFrom='Meerut'
stationFromCoordinates={"lat":"12.12.54.4","long":"44.36.09"} # Meerut
stationToCoordinates={'lat':'54.45.56',"long":'45.45.45'}
currentCoordinates={'lat': '28.98', 'long': '77.7064'}
stationTo='Delhi'
reachedDestination=False




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

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

yelloLine = [270, 0, 270, 600]

RedLine = [173, 0, 173, 600]

totalCountUp = []
#mask=cv2.imread('mask.jpg')
entry_count = 0
def main():
    while True:
        
        

        success, img = cap.read()
        # imgRegion=cv2.bitwise_and(img,mask)

        results = model(img, stream=True)

        detections = np.empty((0, 5))

        for r in results:
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

        resultsTracker = tracker.update(detections)

        cv2.line(img, (yelloLine[0], yelloLine[1]), (yelloLine[2], yelloLine[3]), (0, 0, 255), 5)
        cv2.line(img, (RedLine[0], RedLine[1]), (RedLine[2], RedLine[3]), (0, 255, 200), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                            scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if yelloLine[0] - 20 < cx < yelloLine[2] + 20:
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    dict[id] = [False]
                    cv2.line(img, (yelloLine[0], yelloLine[1]), (yelloLine[2], yelloLine[3]), (0, 0, 255), 5)
                elif totalCountUp.count(id) == 1:
                    if (dict[id].count(False) < 1):
                        dict[id].append(False)
                    cv2.line(img, (yelloLine[0], yelloLine[1]), (yelloLine[2], yelloLine[3]), (0, 0, 255), 5)
            if RedLine[0] - 20 < cx < RedLine[2] + 30:
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    dict[id] = [True]
                    dict[id].append(currentCoordinates)
                    #adding timestamp
                    current_time = datetime.datetime.now()
                    dict[id].append(current_time)
                    cv2.line(img, (RedLine[0], RedLine[1]), (RedLine[2], RedLine[3]), (0, 255, 200), 5)
                elif totalCountUp.count(id) == 1:
                    if (dict[id].count(True) < 1):
                        dict[id].append(True)
                        dict[id].append(currentCoordinates)
                        #adding timestamp
                        current_time = datetime.datetime.now()
                        dict[id].append(current_time)
                        
                        cv2.line(img, (RedLine[0], RedLine[1]), (RedLine[2], RedLine[3]), (0, 255, 200), 5)

        print(totalCountUp)
        entry_count=0

        for i in dict.values():
            if (len(i) >= 2):
                if i[0] == True and i[1] == False:
                    print('in true/False')
                    if entry_count > 0:
                        entry_count -= 1
                    
                if i[0] == False and i[1] == True:
                    print('in /False/True')
                    entry_count += 1
                    
            
        
        # print('count is ', entry_count)
        # print(dict)

        
        cv2.putText(img, str(entry_count), (110, 245), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
        print('count is ', entry_count)
        print(dict)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        print(entry_count)
        
        # if currentCoordinates==stationToCoordinates:
        #     reachedDestination=True
        #     fareCalculations()
        #     dict=[]
        # break
        if entry_count>2:
            print('reached destination')
            fareCalculations()
            break
while True:
    main()
    