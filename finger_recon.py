import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib  # 모델 저장 및 로드용 라이브러리

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# KNN 모델 로드
knn = joblib.load('knn_finger_spelling_model.pkl')  # 사전에 학습된 모델 로드

def recognize_finger_spelling(frame):
    # MediaPipe를 통해 손 관절을 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 관절 좌표 추출
            joints = []
            for lm in hand_landmarks.landmark:
                joints.append([lm.x, lm.y, lm.z])

            # KNN 모델을 사용하여 지문자 예측
            joints = np.array(joints).flatten().reshape(1, -1)
            probabilities = knn.predict_proba(joints)
            
            # 최대 확률 값을 유사도로 사용
            max_probability = np.max(probabilities) * 100  # 퍼센트로 변환
            rounded_similarity = 10 * round(max_probability / 10)  # 10% 간격으로 반올림
            return rounded_similarity  # 반올림된 유사도 반환
    return 0  # 손이 인식되지 않으면 0 반환
