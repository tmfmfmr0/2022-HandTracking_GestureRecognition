import cv2
import mediapipe as mp
import numpy as np
import time
import math

class handDetector() :
	def __init__(self, mode=False, maxHands=1, detectionConfidence=0.5, trackingConfidence=0.5) :
		self.mode = mode
		self.maxHands = maxHands
		self.detectionConfidence = detectionConfidence
		self.trackingConfidence = trackingConfidence
		
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands()
		self.mpDraw = mp.solutions.drawing_utils
		self.tipId = [4,8,12,16,20] # 엄지, 검지, 중지, 약지, 소지 끝

	def findHands(self, img, draw=True) :
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imgRGB)

		if self.results.multi_hand_landmarks :
			for handLM in self.results.multi_hand_landmarks :
				if draw :
					self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)
		return img
	
	def findPosition(self, img, handNo=0, draw=True) :
		self.lmList = [] # 랜드마크 리스트
		
		if self.results.multi_hand_landmarks :
			myHand = self.results.multi_hand_landmarks[handNo]
			
			for id, lm in enumerate(myHand.landmark) :
				#print(id, lm)
				h, w, c = img.shape
				cx, cy= int(lm.x * w), int(lm.y * h)
				self.lmList.append([id,cx,cy])
				# if draw :
				#     cv2.circle(img, (cx, cy), 7, (255,0,255), cv2.FILLED)
					
		return self.lmList
	
	def findDistance(self, p1, p2, img, draw=True, r=7, t=2) :
		x1, y1 = self.lmList[p1][1:]
		x2, y2 = self.lmList[p2][1:]
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
		
		if draw:
			cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
			cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
			cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
			cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
			length = math.hypot(x2 - x1, y2 - y1)
			
		return length, img, [x1, y1, x2, y2, cx, cy]
	
	def fingersOpen(self) :
		fingers = []
		if self.lmList[self.tipId[0]][1] > self.lmList[self.tipId[0]-1][1] : # 엄지
			fingers.append(1)
		else :
			
			fingers.append(0)
		for id in range(1,5) : # 엄지제외
			if self.lmList[self.tipId[id]][2] < self.lmList[self.tipId[id]-2][2] : 
				fingers.append(1)
			else :
				fingers.append(0)
				
		return fingers

def main() :
	pTime = 0
	cTime = 0
	cap = cv2.VideoCapture(0)
	detector = handDetector()

	while True:
		success, img = cap.read()
		img = detector.findHands(img)
		lmList = detector.findPosition(img)
		if len(lmList) != 0 :
			print(lmList[4], lmList[8])
		
		cTime = time.time()
		fps = 1 / (cTime - pTime)
		pTime = cTime
		
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
		cv2.imshow("Image", img)
		if cv2.waitKey(1) & 0xff == ord('q') :
			break

if __name__ == "__main__":
	main()

