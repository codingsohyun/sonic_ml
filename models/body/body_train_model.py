import numpy as np
import os
import tensorflow as tf
from model_architecture import create_lstm_model, load_features_and_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 데이터셋 경로
dataset_path = r'D:\sonic_ml\raw_dataset\words'

# 데이터 로드
print("Loading data...")
features, labels = load_features_and_labels(dataset_path)

# 시퀀스 패딩 (각 시퀀스의 길이를 동일하게 맞춤)
max_sequence_length = max(len(f) for f in features)
padded_features = tf.keras.preprocessing.sequence.pad_sequences(features, maxlen=max_sequence_length, padding='post', dtype='float32')

# 레이블을 숫자로 변환하고 원-핫 인코딩
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(integer_encoded_labels)

# 학습셋과 테스트셋 분리
X_train, X_test, y_train, y_test = train_test_split(padded_features, categorical_labels, test_size=0.2, random_state=42)

# 모델 생성
input_shape = (max_sequence_length, padded_features.shape[2])  # 시퀀스 길이와 특징 차원
num_classes = len(np.unique(labels))
model = create_lstm_model(input_shape, num_classes)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 모델 저장
model.save('lstm_sign_language_model.h5')
print("Model training complete and saved as 'lstm_sign_language_model.h5'")

# 학습 기록 저장 (옵션)
np.save('train_history.npy', history.history)
print("Training history saved as 'train_history.npy'")
