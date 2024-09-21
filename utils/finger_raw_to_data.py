import cv2
import mediapipe as mp
import numpy as np
import os
from data_augmentation import augment_video_frames  # 추가

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def augment_data(frames):
    augmented_data = [frames]  # 원본 데이터 추가
    for aug_frames in augment_video_frames(frames, num_augments=5):  # 증강된 프레임 추가
        augmented_data.append(aug_frames)
    
    # 차원을 맞춘 후 결합
    return np.concatenate(augmented_data, axis=0)

def augment_joint_data(joints):
    augmented_joints = [joints]  # 원본 데이터 추가
    for _ in range(5):
        # 조인트 데이터를 프레임에 맞게 증강 (예: 좌우 반전)
        augmented_joints.append(joints * -1)  # 단순 예시: 좌우 반전
    return np.concatenate(augmented_joints, axis=0)

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        frames = []  # 각 프레임 저장
        joints_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임이 None이 아닌지 확인
            if frame is None:
                print(f"Warning: Skipping empty frame in video {video_path}")
                continue

            # 이미지 채널 수 확인 후, 흑백이면 3채널로 변환
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # OpenCV BGR을 RGB로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    joints = []
                    for lm in hand_landmarks.landmark:
                        joints.append([lm.x, lm.y, lm.z])
                    joints_data.append(joints)
            frames.append(image)

        cap.release()
        joints_data = np.array(joints_data)

        # 데이터 증강 적용
        frames = augment_data(frames)  # 데이터 증강 후 프레임 처리
        joints_data = augment_joint_data(joints_data)  # 데이터 증강 후 조인트 데이터 처리
        
        return joints_data

    except cv2.error as e:
        print(f"OpenCV error processing video {video_path}: {e}")
        return np.array([])
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return np.array([])

def save_data_for_all_classes(video_dir):
    all_data = []
    all_labels = []

    # 각 클래스별로 폴더를 순회하며 데이터를 처리
    for class_dir in sorted(os.listdir(video_dir)):
        class_path = os.path.join(video_dir, class_dir)
        
        if os.path.isdir(class_path):  # 폴더인지 확인
            try:
                label = int(class_dir)  # 폴더 이름을 레이블로 변환
            except ValueError:
                print(f"Skipping non-numeric folder: {class_dir}")
                continue

            print(f"Processing class: {class_dir}, label: {label}")
            
            for video_file in filter(lambda f: f.endswith('.mp4'), os.listdir(class_path)):  # .mp4 파일만 처리
                video_path = os.path.join(class_path, video_file)
                joints_data = process_video(video_path)

                if joints_data.size > 0:
                    all_data.extend(joints_data)
                    all_labels.extend([label] * len(joints_data))

    return np.array(all_data), np.array(all_labels)

# 이 블록을 추가해 파일을 직접 실행할 때만 데이터셋을 처리하도록 함
if __name__ == "__main__": 
    video_dir = '/mnt/8TB_2/sohyun/sonic/sonic_ml/raw_dataset/letters'  # 각 클래스 폴더가 위치한 경로
    data, labels = save_data_for_all_classes(video_dir)

    output_dir = 'D:/sonic_ml/outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'hand_joint_data.npy'), data)
    np.save(os.path.join(output_dir, 'hand_labels.npy'), labels)
