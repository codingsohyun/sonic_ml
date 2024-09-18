import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 데이터 로드 함수
def load_features_and_labels(dataset_dir):
    features = []
    labels = []

    word_classes = os.listdir(dataset_dir)
    for word_class in word_classes:
        class_dir = os.path.join(dataset_dir, word_class)
        if not os.path.isdir(class_dir):
            continue

        # 클래스 폴더에서 특징(.npy) 파일들 로드
        feature_files = [f for f in os.listdir(class_dir) if f.endswith('_features.npy')]
        for feature_file in feature_files:
            feature_path = os.path.join(class_dir, feature_file)
            feature_data = np.load(feature_path, allow_pickle=True)
            
            # 특징 데이터와 해당 레이블(클래스) 저장
            features.append(feature_data)
            labels.append(word_class)

    return np.array(features), np.array(labels)

# LSTM 모델 생성 함수
def create_lstm_model(input_shape, num_classes):
    model = Sequential()

    # LSTM 레이어 (특징의 시퀀스를 학습)
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))

    # 출력 레이어
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# 메인 실행 부분
if __name__ == "__main__":
    dataset_path = r'D:\sonic_ml\raw_dataset\words'

    # 특징과 레이블 로드
    features, labels = load_features_and_labels(dataset_path)

    # 각 시퀀스의 길이를 맞추기 위한 패딩 (최대 길이에 맞추기)
    max_sequence_length = max(len(f) for f in features)
    padded_features = tf.keras.preprocessing.sequence.pad_sequences(features, maxlen=max_sequence_length, padding='post', dtype='float32')

    # 레이블을 숫자로 변환
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # 모델 저장
    model.save('lstm_sign_language_model.h5')

    print("Model training complete and saved as 'lstm_sign_language_model.h5'")
