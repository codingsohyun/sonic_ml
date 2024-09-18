import numpy as np
import tensorflow as tf
from model_architecture import create_lstm_model, load_features_and_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.utils import to_categorical

# 하이퍼파라미터 튜닝을 위한 모델 생성 함수
def create_model(lstm_units=128, learning_rate=0.001):
    model = create_lstm_model((100, 66), num_classes=10)  # 여기서 num_classes는 실제 데이터에 맞게 설정
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_path = r'D:\sonic_ml\raw_dataset\words'

    # 데이터 로드
    features, labels = load_features_and_labels(dataset_path)

    # 시퀀스 패딩
    max_sequence_length = max(len(f) for f in features)
    padded_features = tf.keras.preprocessing.sequence.pad_sequences(features, maxlen=max_sequence_length, padding='post', dtype='float32')

    # 레이블 변환
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(integer_encoded_labels)

    # 학습 및 검증셋 분리
    X_train, X_test, y_train, y_test = train_test_split(padded_features, categorical_labels, test_size=0.2, random_state=42)

    # KerasClassifier를 사용하여 모델 래핑
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # 하이퍼파라미터 범위 설정
    param_dist = {
        'lstm_units': [64, 128, 256],
        'learning_rate': [0.001, 0.01, 0.0001],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20, 30]
    }

    # RandomizedSearchCV를 사용하여 하이퍼파라미터 탐색
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, cv=3)
    random_search.fit(X_train, y_train)

    # 최적의 하이퍼파라미터 출력
    print(f"Best parameters found: {random_search.best_params_}")
