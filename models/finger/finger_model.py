import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = 'D:/sonic_ml/outputs'
data = np.load(os.path.join(output_dir, 'hand_joint_data.npy'))
labels = np.load(os.path.join(output_dir, 'hand_labels.npy'))

data = data.reshape(data.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# KNN 모델 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

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

confusion_matrix_image_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 학습된 모델 저장
joblib.dump(knn, os.path.join(output_dir, 'knn_finger_spelling_model.pkl'))
