import cv2
import mediapipe as mp
import time     # 프레임 레이트 계산 용도

cap = cv2.VideoCapture(0)           # 웹캠 비디오 오브젝트

mpHands = mp.solutions.hands        # hands.py Hands()클래스 손 오브젝트 생성
hands = mpHands.Hands()             # Hands()의 패러미터 : static_image_mode, max_num_hands, 
                                    #              min_detection_confidence, min_tracking_confidence
mpDraw = mp.solutions.drawing_utils # drawing_utils.py의 랜드마크 표시 도구

pTime = 0   # fps 계산을 위한 시간 초기화
cTime = 0

while True:
    success, img = cap.read()   # 웹캠 이미지 프레임
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # bgr to rgb 변환
    results = hands.process(imgRGB)     # hands 오브젝트에 rgb 이미지 전달 (rgb 이미지만 사용)

    if results.multi_hand_landmarks:    # 손 랜드마크 인식 되면
        for handLM in results.multi_hand_landmarks:    # 인식되는 각각의 손마다
            for id, lm in enumerate(handLM.landmark):      # 각각의 랜드마크 표시
                #print(id, lm)          # 랜드마크 각각의 아이디, x y z 좌표
                h, w, c = img.shape     # 높이 너비 채널
                cx, cy = int(lm.x * w), int(lm.y * h)   # x y 좌표 계산
                #print(id, cx, cy)      # 랜드마크별 계산된 x좌표, y좌표 출력

                #if id == 4 :           # 특정 랜드마크 표시 설정
                cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)   # 각각의 손마다 21개의 랜드마크, 커넥션 그리기

    cTime = time.time()     # fps 계산
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)  # fps 표시

    cv2.imshow("Image", img)    # 웹캠 윈도우
    if cv2.waitKey(1) & 0xff == ord('q'):   # q 입력 시 종료
        break
    