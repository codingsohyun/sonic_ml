import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib  # 모델 저장을 위한 라이브러리

# 지문자 영상에서 추출한 손 관절 좌표 데이터 불러오기
data = np.load('hand_joint_data.npy')
labels = np.load('hand_labels.npy')

# 학습 및 테스트 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# KNN 모델 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 테스트 데이터로 정확도 평가
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 학습된 모델 저장
joblib.dump(knn, 'knn_finger_spelling_model.pkl')
