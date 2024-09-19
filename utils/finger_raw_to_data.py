import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def augment_data(joints_data):
    augmented_data = []

    # 원본 데이터 추가
    augmented_data.append(joints_data)

    # 좌우 반전 데이터 추가
    flipped_joints_data = joints_data.copy()
    for frame in flipped_joints_data:
        for joint in frame:
            joint[0] = 1.0 - joint[0]  # X 좌표만 반전
    augmented_data.append(flipped_joints_data)

    # 다른 data augmentation 필요하면.. 더 추가하기!

    return np.concatenate(augmented_data, axis=0)

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
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
        joints_data = np.array(joints_data)

        # 데이터 증강 적용
        joints_data = augment_data(joints_data)
        
        return joints_data

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return np.array([]) 

def save_data_for_all_classes(video_dir):
    all_data = []
    all_labels = []

    # 각 클래스별로 폴더를 순회하며 데이터를 처리
    for label, class_dir in enumerate(sorted(os.listdir(video_dir))):
        class_path = os.path.join(video_dir, class_dir)
        
        if os.path.isdir(class_path):  # 폴더인지 확인
            print(f"Processing class: {class_dir}, label: {label}")
            
            for video_file in os.listdir(class_path):
                video_path = os.path.join(class_path, video_file)
                joints_data = process_video(video_path)

                if joints_data.size > 0:
                    all_data.extend(joints_data)
                    all_labels.extend([label] * len(joints_data))

    return np.array(all_data), np.array(all_labels)

# 데이터셋 처리 및 저장
video_dir = 'D:/sonic_ml/sonic/raw_dataset/letters'  # 각 클래스 폴더가 위치한 경로
data, labels = save_data_for_all_classes(video_dir)

output_dir = 'D:/sonic_ml/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(os.path.join(output_dir, 'hand_joint_data.npy'), data)
np.save(os.path.join(output_dir, 'hand_labels.npy'), labels)
