import cv2
import mediapipe as mp
import numpy as np
import os
from data_augmentation import augment_video_frames

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands()
pose = mp_pose.Pose()

def augment_data(frames):
    augmented_data = [frames]  # Original frames
    for aug_frames in augment_video_frames(frames, num_augments=5):
        augmented_data.append(aug_frames)
    return np.concatenate(augmented_data, axis=0)  # Concatenate frames

def augment_joint_data(joints):
    augmented_joints = [joints]  # Original joint data
    for _ in range(5):  # Add variations to the joint data
        noise = np.random.normal(0, 0.01, joints.shape)  # Add noise
        flipped_joints = joints * -1  # Flip the joints
        noisy_joints = joints + noise  # Add noise
        augmented_joints.append(flipped_joints)
        augmented_joints.append(noisy_joints)
    return np.concatenate(augmented_joints, axis=0)  # Concatenate joint data

# 손과 포즈 랜드마크 데이터를 결합하여 99차원 배열을 반환하는 함수
def extract_pose_hand_keypoints(results_hand, results_pose):
    hand_keypoints = []
    pose_keypoints = []

    # 손 랜드마크 처리 (21개 랜드마크)
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            hand_keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    else:
        hand_keypoints = [(0, 0, 0)] * 21  # 감지되지 않으면 0으로 채움

    # 포즈 랜드마크 처리 (33개 랜드마크)
    if results_pose.pose_landmarks:
        pose_keypoints = [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark]
    else:
        pose_keypoints = [(0, 0, 0)] * 33  # 감지되지 않으면 0으로 채움

    # 손과 포즈 랜드마크 데이터를 결합 (21 * 3 + 33 * 3 = 99차원)
    combined_keypoints = np.array(hand_keypoints + pose_keypoints).flatten()  # 1차원으로 변환

    # import pdb
    # pdb.set_trace()
    # 데이터 크기 출력 (99차원이 되는지 확인)
    print(f"Extracted keypoints shape: {combined_keypoints.shape}")  # (99,)로 나와야 함
    return combined_keypoints

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
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hand = hands.process(image)
            results_pose = pose.process(image)

            joints = extract_pose_hand_keypoints(results_hand, results_pose)
            joints_data.append(joints)
            frames.append(image)

        cap.release()

        frames = augment_data(frames)  # Augment frames
        if len(joints_data) > 0:
            joints_data = augment_joint_data(np.array(joints_data))  # Augment joint data
        else:
            print(f"No landmarks found in video {video_path}")

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
