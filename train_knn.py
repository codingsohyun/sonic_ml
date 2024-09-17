import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib  # 모델 저장을 위한 라이브러리

# 지문자 영상에서 추출한 손 관절 좌표 데이터 불러오기
data = np.load('hand_joint_data.npy')
labels = np.load('hand_labels.npy')

# 데이터 평탄화: 각 프레임의 손 관절 좌표를 1차원 벡터로 변환
# 기존 데이터 형식: (비디오의 프레임 수, 손 관절 수, 좌표 수)
# 변경 후 형식: (비디오의 프레임 수, 손 관절 수 * 좌표 수)
data = data.reshape(data.shape[0], -1)

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
