import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(output_dir):
    # 데이터 파일 경로 설정
    data_path = os.path.join(output_dir, 'hand_joint_data.npy')
    labels_path = os.path.join(output_dir, 'hand_labels.npy')

    # 데이터 파일 존재 여부 확인 및 로드
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Data files not found in {output_dir}")

    print(f"Loading data from {output_dir}...")
    data = np.load(data_path)
    labels = np.load(labels_path)

    # 데이터 차원 변경 (2D로 변환)
    data = data.reshape(data.shape[0], -1)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    return data, labels

def tune_knn_model(X_train, y_train):
    # 하이퍼파라미터 튜닝을 위한 GridSearchCV 설정
    print("Tuning KNN model using GridSearchCV...")
    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    knn = KNeighborsClassifier()

    # GridSearchCV를 사용하여 최적의 n_neighbors 찾기
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=2)  # verbose 옵션을 통해 프로세스 출력
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    # 테스트셋 평가
    print("Evaluating the model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # confusion matrix 및 시각화
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    return y_pred

def save_model_and_results(model, output_dir, y_test, y_pred):
    # confusion matrix 이미지 저장
    confusion_matrix_image_path = os.path.join(output_dir, 'confusion_matrix.png')
    print(f"Saving confusion matrix to {confusion_matrix_image_path}...")
    plt.savefig(confusion_matrix_image_path)

    # classification report 출력
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 학습된 모델 저장
    model_path = os.path.join(output_dir, 'knn_finger_spelling_model.pkl')
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":

    output_dir = '/mnt/8TB_2/sohyun/sonic/sonic_ml/outputs'  # 수정된 경로

    # 데이터 로드
    print("Loading data...")
    data, labels = load_data(output_dir)

    # 학습셋과 테스트셋 분리
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # KNN 모델 하이퍼파라미터 튜닝 및 학습
    print("Starting model training...")
    best_knn = tune_knn_model(X_train, y_train)

    # 모델 평가
    print("Model evaluation in progress...")
    y_pred = evaluate_model(best_knn, X_test, y_test)

    # 결과 저장 및 모델 저장
    print("Saving model and results...")
    save_model_and_results(best_knn, output_dir, y_test, y_pred)

    print("Process completed successfully.")
