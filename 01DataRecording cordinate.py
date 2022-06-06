import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os

# 제스처 종류
actions = ['a', 'b', 'c']

# 시퀀스 길이, 녹화시간
seq_length = 10
recording_time = 5

# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 웹캠
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 데이터셋 저장 폴더 생성
os.makedirs('Dataset', exist_ok=True)

# 캠 열려있는동안
while cap.isOpened():
    # 각 동작마다
	for label, action in enumerate(actions):

		angleData = []

		ret, img = cap.read()

		# 녹화 준비 메세지 출력
		cv2.putText(img, f'Record {action} action', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
		cv2.imshow('img', img)
		cv2.waitKey(3000)

		# 녹화 시작
		# 녹화 시간동안
		start_time = time.time()
		while time.time() - start_time < recording_time:
			ret, img = cap.read()

			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			result = hands.process(img)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

			# 손 있을 때
			if result.multi_hand_landmarks:
				for res in result.multi_hand_landmarks:
					# 모든 랜드마크 좌표
					lm_coordinates = np.zeros((21, 3))
					# 각 랜드마크 좌표
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
					mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

			# 화면 출력, q누르면 종료
			cv2.imshow('img', img)
			if cv2.waitKey(1) == ord('q'):
				break

		# print(action, angleData, end='\n\n\n')
		
		# 문제지-정답지 데이터프레임 생성
		df = pd.DataFrame(angleData)
		df['label'] = label

		# print(action, df, end='\n\n\n')

		# 데이터프레임 저장
		df.to_csv(f".\\Dataset\\{action}.csv", header=None, index=None)

	break
