import numpy as np
import os
import tensorflow as tf
from model_architecture import load_features_and_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical

# 데이터셋 경로
dataset_path = r'D:\sonic_ml\raw_dataset\words'

# 학습된 모델 로드
model = tf.keras.models.load_model('lstm_sign_language_model.h5')

# 데이터 로드
print("Loading data for evaluation...")
features, labels = load_features_and_labels(dataset_path)

# 시퀀스 패딩
max_sequence_length = max(len(f) for f in features)
padded_features = tf.keras.preprocessing.sequence.pad_sequences(features, maxlen=max_sequence_length, padding='post', dtype='float32')

# 레이블을 숫자로 변환하고 원-핫 인코딩
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(integer_encoded_labels)

# 학습셋과 테스트셋 분리 (평가할 때도 동일한 방법으로 나눔)
X_train, X_test, y_train, y_test = train_test_split(padded_features, categorical_labels, test_size=0.2, random_state=42)

# 모델 평가
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 세부 성능 지표 (예측값과 실제값을 비교)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 분류 보고서 출력
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# 정확도 출력
print(f"\nOverall Accuracy: {accuracy_score(y_true_classes, y_pred_classes)}")
