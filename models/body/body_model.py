import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class SignLanguageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)  # 시퀀스 길이에 맞게 패딩
    labels = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels

def load_data():
    data_path = '/mnt/8TB_2/sohyun/sonic/sonic_ml/outputs'
    features_file = os.path.join(data_path, 'body_joint_data.npy')
    labels_file = os.path.join(data_path, 'body_labels.npy')
    
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Data files are not found. Please check the path.")
    
    features = np.load(features_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    return features, labels

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 256).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 256).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    input_size = X_train[0].shape[1]  # 조인트 데이터의 크기 설정
    model = LSTMModel(input_size=input_size, hidden_size=256, num_layers=2, num_classes=10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 첫 번째 가중치 확인을 위한 텐서 (lstm 가중치)
    lstm_weight_tensor = model.lstm.weight_ih_l0

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss}')

        # LSTM 첫 번째 레이어의 가중치 일부를 출력
        print(f"Epoch {epoch+1}, LSTM Weight Sample (First Layer): {lstm_weight_tensor[0][:5]}")

        # 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    features, labels = load_data()
    train_model(features, labels)
