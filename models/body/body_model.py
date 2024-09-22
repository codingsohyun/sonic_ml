import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# utils 폴더에서 dataset_utils 모듈을 불러오기 위해 경로 추가
sys.path.append('/mnt/8TB_2/sohyun/sonic/sonic_ml/utils')

from body_dataset_utils import SignLanguageDataset, pad_collate_fn
from model_hyperparameter_tuning import hyperparameter_tuning  # 하이퍼파라미터 튜닝 코드 가져오기

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
    
    # Adjust labels: if labels are between 71-80, shift them to start at 0
    labels = labels - 71
    print(f"Converted labels: {np.unique(labels)}")

    # Ensure features are 2D (sequence_length, feature_size)
    if features.ndim == 1:  # If the entire dataset is 1D
        features = np.expand_dims(features, axis=1)
    elif features[0].ndim == 1:  # If each feature is 1D, expand each to 2D
        features = np.array([np.expand_dims(f, axis=0) for f in features])

    return features, labels


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))  # Pass through LSTM
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

def train_model_with_best_params(features, labels, best_params):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=pad_collate_fn)
    
    input_size = X_train[0].shape[1]  # Set input size based on the feature size (162 for hand + pose landmarks)
    model = LSTMModel(input_size=input_size, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], num_classes=10)  # Set 10 classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(best_params['epochs']):
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

        # Evaluate the model
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

    # Save the trained model to the specified path
    output_model_path = '/mnt/8TB_2/sohyun/sonic/sonic_ml/outputs/new_lstm_sign_language_model.pth'
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved at: {output_model_path}")


if __name__ == "__main__":
    features, labels = load_data()

    # Check if features are loaded correctly and have at least 2 dimensions
    if len(features) == 0:
        raise ValueError("No features loaded. Please check your data.")
    if features[0].ndim == 1:
        raise ValueError("Each feature should be at least 2-dimensional (sequence_length, feature_size)")

    # 하이퍼파라미터 범위 설정
    param_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20]
    }

    input_size = features[0].shape[1]  # Feature size
    num_classes = 10  # Number of classes (0-9)

    # 하이퍼파라미터 튜닝 실행
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    best_params, best_accuracy = hyperparameter_tuning(X_train, y_train, X_test, y_test, param_grid, input_size, num_classes)
    print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")

    # 최적의 하이퍼파라미터로 모델 학습
    train_model_with_best_params(features, labels, best_params)
