import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# KNN 모델 로드 (모델을 학습한 후 저장했다면 여기서 불러오기)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 데이터셋에서 학습된 모델을 로드하세요

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
            prediction = knn.predict(joints)

            return prediction[0]  # 예측 결과 반환
    return 0  # 손이 인식되지 않으면 0 반환
