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
from model_hyperparameter_tuning import hyperparameter_tuning  # 추가된 부분

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

        feature_files = [f for f in os.listdir(class_dir) if f.endswith('_features.npy')]
        for feature_file in feature_files:
            feature_path = os.path.join(class_dir, feature_file)
            feature_data = np.load(feature_path, allow_pickle=True)

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
        out = self.fc(out[:, -1, :])
        return out

# 데이터셋 패딩 함수
def pad_collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels_tensor

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = r'D:/sonic_ml/raw_dataset/words'
    features, labels = load_features_and_labels(dataset_path)

    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)
    features, labels = shuffle(features, integer_encoded_labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    input_size = X_train[0].shape[1] if len(X_train) > 0 else 0
    num_classes = len(np.unique(labels))

    # 하이퍼파라미터 범위 설정
    param_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20]
    }

    # 하이퍼파라미터 튜닝 실행
    best_params, best_accuracy = hyperparameter_tuning(X_train, y_train, param_grid, input_size, num_classes)
    print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")
