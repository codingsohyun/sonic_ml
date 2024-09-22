import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterSampler
import sys
import os

# utils 폴더의 경로 추가
sys.path.append('/mnt/8TB_2/sohyun/sonic/sonic_ml/utils')

from body_dataset_utils import SignLanguageDataset, pad_collate_fn

# LSTM 모델 정의 (body_model.py와 동일)
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

# 평가 함수 (모델 평가용)
def evaluate_model(model, X_test, y_test, device):
    model.eval()
    test_dataset = SignLanguageDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# 하이퍼파라미터 튜닝을 위한 함수
def hyperparameter_tuning(X_train, y_train, X_test, y_test, param_grid, input_size, num_classes):
    best_accuracy = 0
    best_params = None

    # device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 하이퍼파라미터 샘플링
    param_list = list(ParameterSampler(param_grid, n_iter=5))

    for params in param_list:
        print(f"Testing params: {params}")

        # 모델 생성
        model = LSTMModel(input_size, params['hidden_size'], params['num_layers'], num_classes).to(device)

        # 옵티마이저 및 손실 함수
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # 데이터 로더
        train_dataset = SignLanguageDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=pad_collate_fn)

        # 학습 진행
        model.train()
        for epoch in range(params['epochs']):
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 모델 평가
        accuracy = evaluate_model(model, X_test, y_test, device)
        print(f"Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    return best_params, best_accuracy
