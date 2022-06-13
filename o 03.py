import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# 제스처 종류
actions = ['b', 'c', 'd', 'e']
# 데이터 시퀀스 길이, 녹화시간
seqLength = 15
# 모델 load
model = load_model('./Models/model.h5')
# MediaPipe hands model 초기화
mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# 웹캠
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

angle_data = []
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
            coord = np.zeros((21, 3))
            for i, lm in enumerate(res.landmark):
                coord[i] = [lm.x, lm.y, lm.z]

            a = coord[[ 5,  1,2,3,  5,6,7,   9,10,11,  13,14,15,  17,18,19,  0, 0], :]
            b = coord[[17,  2,3,4,  6,7,8,  10,11,12,  14,15,16,  18,19,20,  5,17], :]
            vec = b - a
            vec_unit = vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]
            vec_unit = np.append(vec_unit, [[1,0,0]], axis=0)
            vec_unit = np.append(vec_unit, [[0,1,0]], axis=0)
            vec_unit = np.append(vec_unit, [[0,0,1]], axis=0)
            
            angle = np.arccos( np.einsum('ij, ij->i',
                vec_unit[[0,1,2,  0,4,5,  0,7,8,   0,10,11,   0,13,14,  3,6,9,12,15,  6,9,12,15,   0, 0, 0], :],
                vec_unit[[1,2,3,  4,5,6,  7,8,9,  10,11,12,  13,14,15,  0,0,0, 0, 0,  2,2, 2, 2,  18,19,20], :]))
            angle_data.append(np.degrees(angle))

            # 랜드마크 표시
            mpDrawing.draw_landmarks(img, res, mpHands.HAND_CONNECTIONS)

        ## 판단
            # 설정한 시퀀스길이만큼 데이터 생겨야 판단
            if len(angle_data) < seqLength:
                continue
            # 설정한 시퀀스길이만큼의 데이터를 문제지로
            Xdata = np.expand_dims(np.array(angle_data[-seqLength:]), axis=0)
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

