import numpy as np
import tensorflow as tf
import mediapipe as mp

# LSTM 모델 로드
model = tf.keras.models.load_model('models/body/lstm_sign_language_model.h5')

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints(results):
    """
    Mediapipe 결과로부터 keypoints(관절 좌표)를 추출하여 모델의 입력으로 변환.
    """
    if results.pose_landmarks:
        keypoints = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(99)  # 관절이 감지되지 않았을 때 대체값
    return keypoints

def body_inference(frame):
    """
    주어진 프레임으로부터 포즈 키포인트를 추출하고 LSTM 모델을 통해 예측.
    """
    # 프레임을 RGB로 변환하여 Mediapipe에 입력
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 포즈 키포인트 추출
    keypoints = extract_keypoints(results)
    
    # LSTM 모델의 입력 형태로 변환 (1, 99) -> (1, 1, 99) for batch and sequence length
    keypoints_input = np.expand_dims(np.expand_dims(keypoints, axis=0), axis=0)

    # 모델을 통해 예측 수행
    prediction = model.predict(keypoints_input)

    # 예측 결과에서 가장 높은 확률을 가진 클래스 선택
    predicted_class = np.argmax(prediction)

    # 예측 확률값 (0에서 1 사이의 값을 0에서 100으로 변환)
    similarity = np.max(prediction) * 100

    return similarity, predicted_class
