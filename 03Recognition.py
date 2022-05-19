import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# 제스처 종류
actions = ['a', 'b', 'c']
# 데이터 시퀀스 길이, 녹화시간
seqLength = 10
# 모델 load
model = load_model('./Models/model.h5')
# MediaPipe hands model 초기화
mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# 웹캠
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

angleData = []
actionPredicted = []

# 캠 열려있는동안
while cap.isOpened():
    
    ret, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 손 있을 때
    if result.multi_hand_landmarks :
        for res in result.multi_hand_landmarks:
            # 랜드마크 좌표
            lm_coordinates = np.zeros((21, 3))
            for i, lm in enumerate(res.landmark):
                lm_coordinates[i] = [lm.x, lm.y, lm.z]

            # 벡터를 이용한 랜드마크간 각도 계산
            a1 = lm_coordinates[[0,1,2,3,0,5,6,7,0, 9,10,11, 0,13,14,15, 0,17,18,19], :]
            a2 = lm_coordinates[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
            v = a2 - a1
            # 단위벡터로 표준화 normalize
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]    # 내적할 수 있게 열 벡터로
            # 내적을 이용한 각도 계산 ( a•b = |a||b|cos(Θ) )
            angle = np.arccos(np.einsum('nt,nt->n',     # 내적, cos의 역수
                v[[0,1,2,4,5,6,8, 9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            # 라디안 단위 변환
            angle = np.degrees(angle)
            # 데이터 구성
            angleData.append(angle)

            # 랜드마크 표시
            mpDrawing.draw_landmarks(img, res, mpHands.HAND_CONNECTIONS)

        ## 판단
            # 설정한 시퀀스길이만큼 데이터 생겨야 판단
            if len(angleData) < seqLength:
                continue
            # 설정한 시퀀스길이만큼의 데이터를 문제지로
            Xdata = np.expand_dims(np.array(angleData[-seqLength:]), axis=0)
            # # 판단 확률 출력
            # print(model.predict(Xdata))
            # 라벨별 예측 확률
            Yprobabilities = model.predict(Xdata).squeeze()
            # 가장 확률 높은 라벨
            Yindex = int(np.argmax(Yprobabilities))
            # 그 라벨의 확률
            confidence = Yprobabilities[Yindex]

            # 특정 확률 이상일 때
            if confidence < 0.99:
                continue

            action = actions[Yindex]
            actionPredicted.append(action)

            # 특정 횟수만큼 같은 동작이라고 판단되면
            if len(actionPredicted) < 3:
                continue
            predictedAs = ''
            if actionPredicted[-1] == actionPredicted[-2] == actionPredicted[-3]:
                predictedAs = action

            # 판단 결과 출력
            cv2.putText(img, f'{predictedAs}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    # 화면 출력, q누르면 종료
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

