import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os

actions = ['a', 'b', 'c']	# 0, 1, 2...    # 'back', 'home', 'overview', 'click', 'zoomIn', 'zoomOut'

seq_length = 10        # 한번에 학습시킬 데이터 시퀀스 길이 (window size)
recording_time = 5        # 녹화 시간

mp_hands = mp.solutions.hands    # MediaPipe hands model 초기화
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # 웹캠

created_time = int(time.time())

# 데이터셋 저장 폴더 생성
os.makedirs('Dataset', exist_ok=True)

while cap.isOpened():    # 캠 열려있는동안
	for label, action in enumerate(actions):    # 각 동작마다
		
		data = []

		ret, img = cap.read()

		# 녹화 준비 메세지 출력
		cv2.putText(img, f'Record {action.upper()} action', (10, 50), 
					cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
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

			if result.multi_hand_landmarks:    # 손 있을 때
				for res in result.multi_hand_landmarks:    # 손마다
					# 모든 랜드마크 데이터
					lm_coordinates = np.zeros((21, 3))
					# 각 랜드마크 좌표
					for j, lm in enumerate(res.landmark):
						lm_coordinates[j] = [lm.x, lm.y, lm.z]

					# 벡터를 이용한 랜드마크간 각도 계산
					a1 = lm_coordinates[[0,1,2,3,0,5,6,7,0, 9,10,11, 0,13,14,15, 0,17,18,19], :]
					a2 = lm_coordinates[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
					v = a2 - a1    # [20, 3]
					# 단위벡터로 표준화 normalize
					v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]    # 내적할 수 있게 열 벡터로
					# 내적을 이용한 각도 계산 ( a•b = |a||b|cos(Θ) )
					angle = np.arccos(np.einsum('nt,nt->n',     # 내적, cos의 역수
						v[[0,1,2,4,5,6,8, 9,10,12,13,14,16,17,18],:], 
						v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
					# 라디안 단위 변환
					angle = np.degrees(angle)
					# 데이터 구성
					data.append(angle)

					mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

			cv2.imshow('img', img)
			if cv2.waitKey(1) == ord('q'):
				break
		
		#print(action, data, '\n\n')

		df = pd.DataFrame(data)
		df['label'] = label

		print(action, df, end='\n\n\n')

		df.to_csv(f".\\Dataset\\df_{action}_{created_time}.csv", header=None, index=None)

	break
