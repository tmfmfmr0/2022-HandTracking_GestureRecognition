import cv2
import HandTrackingModule as htm
import numpy as np
import time
import autopy

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)

wScreen, hScreen = autopy.screen.size()
frameR = 150 # 프레임-frameR = 사각형
smoothening = 4 # 떨림보정
prevX, prevY = 0, 0
currX, currY = 0, 0

pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)   # findHands
    lmList = detector.findPosition(img, draw=False) # 위치
    
    if len(lmList) != 0 : 
        x1, y1 = lmList[8][1:]	# 검지좌표
        x2, y2 = lmList[12][1:]	# 중지좌표
        # print(lmList[4], lmList[8])
        
        fingers = detector.fingersOpen()
        # print(fingers)
        
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255,100,255), 2)
        
        clickLength, img, lineinfo = detector.findDistance(4,10,img)
        print(clickLength)
        try:
            # 이동모드
            if fingers[1]==1 and clickLength>30:
                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
                y3 = np.interp(y1, (frameR, wCam-frameR), (0, wScreen))
                currX = prevX + (x3-prevX)/smoothening
                currY = prevY + (y3-prevY)/smoothening
                autopy.mouse.move(currX,currY)
                cv2.circle(img, (x1, y1), 15, (255,0,255), cv2.FILLED)
                prevX, prevY = currX, currY
            # 클릭모드
            if fingers[1]==1 and clickLength<30 :
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 7, (0,255,255), cv2.FILLED)
                autopy.mouse.click()
        except:
            pass
            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 3)
    
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xff == ord('q') :
        break