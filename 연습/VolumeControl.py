import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector()   # htm클래스 오브젝트 default 설정함

devices = AudioUtilities.GetSpeakers()  # pycaw 오브젝트
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)   # findHands
    lmList = detector.findPosition(img, draw=False) # 위치
    
    # 랜드마크 위치: 0.손바닥시작, 4.엄지끝, 8.검지끝, 12.중지끝, 20.새끼끝
    if len(lmList) != 0 : 
        #print(lmList[4], lmList[8])
        
        x1, y1 = lmList[4][1], lmList[4][2]     # 엄지
        x2, y2 = lmList[8][1], lmList[8][2]     # 검지
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)
        
        length = math.hypot(x2-x1, y2-y1)
        #print(length)
        
        # hand range 40 ~ 250
        # volume range -65 ~ 0
        vol = np.interp(length, [40, 250], [minvol, maxvol])    # numpy 수치 비례 변환
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length < 50 :    # 버튼 효과 (색변경)
            cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_COMPLEX,
                1, (255,0,255), 3)
    
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xff == ord('q') :
        break