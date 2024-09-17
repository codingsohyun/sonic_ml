import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    joints_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joints = []
                for lm in hand_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
                joints_data.append(joints)

    cap.release()
    return np.array(joints_data)

def save_data_for_all_classes(video_dir):
    all_data = []
    all_labels = []

    # 각 클래스별로 폴더를 순회하며 데이터를 처리
    for label, class_dir in enumerate(os.listdir(video_dir)):
        class_path = os.path.join(video_dir, class_dir)
        
        if os.path.isdir(class_path):  # 폴더인지 확인
            print(f"Processing class: {class_dir}, label: {label}")
            
            for video_file in os.listdir(class_path):
                video_path = os.path.join(class_path, video_file)
                joints_data = process_video(video_path)

                all_data.extend(joints_data)
                all_labels.extend([label] * len(joints_data))

    return np.array(all_data), np.array(all_labels)

# 데이터셋 처리 및 저장
video_dir = 'D:/sonic_ml/sonic/raw_dataset/' 
data, labels = save_data_for_all_classes(video_dir)

np.save('hand_joint_data.npy', data)
np.save('hand_labels.npy', labels)
