import cv2
import mediapipe as mp
import numpy as np
import joblib  # 모델 저장 및 로드용 라이브러리
import os

# 미디어파이프 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 사전에 학습된 모델 로드 (전체 경로로 변경)
model_path = 'D:/sonic_ml/outputs/knn_finger_spelling_model.pkl'
if os.path.exists(model_path):
    knn = joblib.load(model_path)
else:
    raise FileNotFoundError(f"The model file was not found at {model_path}")


def finger_inference(frame):
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
            
            # 손가락 마디마다의 정확도로 코드 수정할 수도..?
            max_probability = np.max(probabilities) * 100  
            rounded_similarity = 10 * round(max_probability / 10) 
            return rounded_similarity  
    return 0 
