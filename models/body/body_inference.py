import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import cv2  # OpenCV 임포트

# PyTorch LSTM 모델 정의 (body_model.py와 일치)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# PyTorch로 학습된 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 경로 수정 (모델 저장 위치와 동일하게 설정)
model_path = 'D:/sonic_ml/outputs/new_lstm_sign_language_model.pth'

# input_size, hidden_size, num_layers와 num_classes는 학습된 모델에 맞춰 설정
input_size = 162  # 162 for hand + pose landmarks
hidden_size = 128  # 학습된 모델에서 hidden_size=128로 설정됨
num_layers = 2     # 학습된 모델에서 num_layers=2로 설정됨
num_classes = 10   # 10개의 클래스 (0-9)

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints(results):
    if results.pose_landmarks:
        # 관절 좌표를 [x, y, z] 형태로 추출한 후 평탄화
        keypoints = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        # 관절이 감지되지 않았을 때는 0으로 채워진 배열 반환
        keypoints = np.zeros(99)  # 관절이 감지되지 않았을 때 대체값
    return keypoints

def body_inference(frame):
    # 프레임을 RGB로 변환하여 Mediapipe에 입력
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 포즈 키포인트 추출
    keypoints = extract_keypoints(results)
    
    # PyTorch 모델의 입력 형태로 변환 (1, 99) -> (1, 1, 99) for batch and sequence length
    keypoints_input = np.expand_dims(np.expand_dims(keypoints, axis=0), axis=0)
    keypoints_input = torch.tensor(keypoints_input, dtype=torch.float32).to(device)

    # 모델을 통해 예측 수행
    with torch.no_grad():
        prediction = model(keypoints_input)

    # 예측 결과에서 가장 높은 확률을 가진 클래스 선택
    predicted_class = torch.argmax(prediction, dim=1).item()

    # 예측 확률값 (0에서 1 사이의 값을 0에서 100으로 변환)
    similarity = torch.softmax(prediction, dim=1).max().item() * 100

    return similarity, predicted_class
