import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os

# 제스처 종류
actions = ['a']

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

		coordData = []

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
					lm_coordinates = []
					# 각 랜드마크 좌표
					for i, lm in enumerate(res.landmark):
						lm_coordinates.append([lm.x, lm.y, lm.z])

					# 데이터 구성
					coordData.append(lm_coordinates)

					# 랜드마크 표시
					mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

			# 화면 출력, q누르면 종료
			cv2.imshow('img', img)
			if cv2.waitKey(1) == ord('q'):
				break

		# 데이터프레임 생성
		df = pd.DataFrame(coordData)
		df['label'] = label

		print(action, df, end='\n\n\n')

		# 데이터프레임 저장
		# df.to_csv(f".\\Dataset\\{action}_coordinate.csv", header=None, index=None)

		# 데이터프레임 넘파이화
		a = df.to_numpy()
		print(a)

	break
