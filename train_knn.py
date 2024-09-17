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

# 여러 개의 n_neighbors 값으로 실험
best_accuracy = 0
best_k = 0
for k in range(1, 16):  # 1부터 15까지의 k 값을 실험
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'k={k}, Accuracy: {accuracy * 100:.2f}%')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        joblib.dump(knn, 'knn_finger_spelling_model.pkl')  # 가장 좋은 모델만 저장

print(f'Best k={best_k} with Accuracy: {best_accuracy * 100:.2f}%')
