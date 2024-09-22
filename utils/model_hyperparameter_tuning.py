import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import ParameterSampler
from torch.nn.utils.rnn import pad_sequence

# SignLanguageDataset 정의 (body_model.py와 동일)
class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 데이터 로드 함수 (body_model.py와 동일)
def load_features_and_labels(dataset_dir):
    features = []
    labels = []

    word_classes = os.listdir(dataset_dir)
    for word_class in word_classes:
        class_dir = os.path.join(dataset_dir, word_class)
        if not os.path.isdir(class_dir):
            continue

        feature_files = [f for f in os.listdir(class_dir) if f.endswith('_features.npy')]
        for feature_file in feature_files:
            feature_path = os.path.join(class_dir, feature_file)
            feature_data = np.load(feature_path, allow_pickle=True)

            features.append(torch.tensor(feature_data, dtype=torch.float32))
            labels.append(word_class)

    return features, labels

# LSTM 모델 정의 (body_model.py와 동일)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 평가 함수 (모델 평가용)
def evaluate_model(model, X_test, y_test):
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

# 데이터셋 패딩 함수
def pad_collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels_tensor

# 하이퍼파라미터 튜닝을 위한 함수
def hyperparameter_tuning(X_train, y_train, param_grid, input_size, num_classes):
    best_accuracy = 0
    best_params = None

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
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    return best_params, best_accuracy

# # 하이퍼파라미터 탐색 실행
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     dataset_path = r'D:/sonic_ml/raw_dataset/words'
#     features, labels = load_features_and_labels(dataset_path)
#     features, labels = shuffle(features, labels)

#     # 레이블을 숫자로 변환
#     label_encoder = LabelEncoder()
#     integer_encoded_labels = label_encoder.fit_transform(labels)

#     # 학습셋과 테스트셋 분리
#     X_train, X_test, y_train, y_test = train_test_split(features, integer_encoded_labels, test_size=0.2, random_state=42)

#     input_size = X_train[0].shape[1] if len(X_train) > 0 else 0  # 특징의 차원 수
#     num_classes = len(np.unique(labels))

#     # 하이퍼파라미터 범위 설정
#     param_grid = {
#         'hidden_size': [64, 128, 256],
#         'num_layers': [1, 2, 3],
#         'learning_rate': [0.001, 0.0001],
#         'batch_size': [16, 32, 64],
#         'epochs': [10, 20]
#     }

#     best_params, best_accuracy = hyperparameter_tuning(X_train, y_train, param_grid, input_size, num_classes)
#     print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")
