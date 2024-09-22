import cv2
import mediapipe as mp
import numpy as np
import os
from data_augmentation import augment_video_frames

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def augment_data(frames):
    augmented_data = [frames]  # 원본 프레임 추가
    for aug_frames in augment_video_frames(frames, num_augments=5):
        augmented_data.append(aug_frames)
    return np.concatenate(augmented_data, axis=0)  # 프레임 차원 결합

def augment_joint_data(joints):
    augmented_joints = [joints]  # 원본 조인트 데이터 추가
    for _ in range(5):  # 다양한 조인트 변형 추가
        noise = np.random.normal(0, 0.01, joints.shape)  # 조인트에 작은 노이즈 추가
        flipped_joints = joints * -1  # 좌우 반전
        noisy_joints = joints + noise  # 노이즈 추가된 조인트
        augmented_joints.append(flipped_joints)
        augmented_joints.append(noisy_joints)
    return np.concatenate(augmented_joints, axis=0)  # 모든 변형된 데이터를 결합

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        frames = []
        joints_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                print(f"Warning: Skipping empty frame in video {video_path}")
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 흑백 확인 없이 바로 RGB 변환
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    joints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    joints_data.append(joints)
            frames.append(image)

        cap.release()

        frames = augment_data(frames)  # 프레임 데이터 증강
        if len(joints_data) > 0:
            joints_data = augment_joint_data(np.array(joints_data))  # 조인트 데이터 변형
        else:
            print(f"No hand landmarks found in video {video_path}")

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

    for class_dir in sorted(os.listdir(video_dir)):
        class_path = os.path.join(video_dir, class_dir)
        if os.path.isdir(class_path):
            try:
                label = int(class_dir)
                print(f"Processing class: {class_dir}, label: {label}")
                videos = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
                if not videos:
                    print(f"No video files found in class directory: {class_dir}")
                for video in videos:
                    video_path = os.path.join(class_path, video)
                    joints_data = process_video(video_path)
                    if joints_data.size > 0:
                        all_data.extend(joints_data)
                        all_labels.extend([label] * len(joints_data))
            except ValueError:
                print(f"Skipping non-numeric folder: {class_dir}")
            except Exception as e:
                print(f"Error processing class {class_dir}: {e}")

    return np.array(all_data), np.array(all_labels)

if __name__ == "__main__":
    video_dir = '/mnt/8TB_2/sohyun/sonic/sonic_ml/raw_dataset/words'
    data, labels = save_data_for_all_classes(video_dir)

    output_dir = '/mnt/8TB_2/sohyun/sonic/sonic_ml/outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'body_joint_data.npy'), data)
    np.save(os.path.join(output_dir, 'body_labels.npy'), labels)
