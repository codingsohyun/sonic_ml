import cv2
import numpy as np
import tensorflow as tf
from feature_extraction import extract_features_from_frame
from tensorflow.keras.preprocessing.sequence import pad_sequences
from visualize_results import visualize_results
from feedback_system import feedback_system

# 학습된 LSTM 모델 로드
model = tf.keras.models.load_model('lstm_sign_language_model.h5')

# 실시간 추론을 위한 전처리 함수
def preprocess_features(features, max_sequence_length):
    features_sequence = [features]
    padded_sequence = pad_sequences(features_sequence, maxlen=max_sequence_length, padding='post', dtype='float32')
    return np.array(padded_sequence)

# 실시간 추론 함수
def real_time_inference(target_class):
    cap = cv2.VideoCapture(0)
    max_sequence_length = 100  # 학습할 때 사용한 최대 시퀀스 길이로 설정

    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임에서 특징 추출
        features = extract_features_from_frame(frame)
        if features:
            sequence.append(features)

        # 충분한 시퀀스가 쌓이면 모델에 입력하여 예측
        if len(sequence) == max_sequence_length:
            input_sequence = preprocess_features(sequence, max_sequence_length)
            prediction = model.predict(input_sequence)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = "some_class_name"  # 예측된 클래스 이름 매핑 필요
            confidence = np.max(prediction)

            # 예측 결과 시각화
            frame = visualize_results(frame, predicted_class, confidence)

            # 피드백 제공
            feedback = feedback_system(predicted_class, target_class)
            print(feedback)

            sequence = []  # 새로운 시퀀스를 위해 초기화

        # 웹캠 화면 출력
        cv2.imshow('Webcam Stream with Predictions', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    target_class = "hello"  # 사용자가 목표로 하는 수어 단어
    real_time_inference(target_class)
