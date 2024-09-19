import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils import shuffle

# 데이터셋 정의
class SignLanguageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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
            features.append(torch.tensor(feature_data, dtype=torch.float32))
            labels.append(word_class)

    return features, labels

# LSTM 모델 정의
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
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력을 사용
        return out

# 데이터셋 패딩 함수
def pad_collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels_tensor

# 메인 실행 부분
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = r'D:/sonic_ml/raw_dataset/words'

    # 특징과 레이블 로드
    features, labels = load_features_and_labels(dataset_path)

    # 레이블을 숫자로 변환
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)
    labels_tensor = torch.tensor(integer_encoded_labels, dtype=torch.long)

    # 학습셋과 테스트셋 분리
    features, labels = shuffle(features, integer_encoded_labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # DataLoader 생성
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    # 모델 생성
    input_size = X_train[0].shape[1]  # 특징 차원 수
    hidden_size = 128
    num_layers = 2
    num_classes = len(np.unique(labels))

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 테스트
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')

    # 모델 저장
    torch.save(model.state_dict(), 'lstm_sign_language_model.pth')
    print("yeahhhh")
