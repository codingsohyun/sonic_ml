import cv2
import mediapipe as mp
import os
import numpy as np
from data_augmentation import augment_video_frames  # 추가

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_features_from_video(video_path, augment=False, num_augments=5):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        frame_list = []
        features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임이 None이 아닌지 확인
            if frame is None:
                print(f"Warning: Skipping empty frame in video {video_path}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)  # 증강을 위해 원본 프레임 저장

        cap.release()
        cv2.destroyAllWindows()

        # 데이터 증강 여부에 따라 증강된 프레임 생성
        if augment:
            augmented_frames = augment_video_frames(frame_list, num_augments=num_augments)
            frame_list = [frame for frames in augmented_frames for frame in frames]  # 4D 차원 맞추기

        # 프레임별 특징 추출
        for frame_rgb in frame_list:
            hand_results = hands.process(frame_rgb)
            pose_results = pose.process(frame_rgb)
            frame_features = {}

            # 손 특징 추출
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_data = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                    frame_features['hand'] = hand_data

            # 신체 특징 추출
            if pose_results.pose_landmarks:
                pose_data = [(landmark.x, landmark.y, landmark.z) for landmark in pose_results.pose_landmarks.landmark]
                frame_features['pose'] = pose_data

            if frame_features:
                features.append(frame_features)

        return features

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return []  # 오류 발생 시 빈 리스트 반환

def process_dataset(dataset_dir, augment=False, num_augments=5):
    try:
        word_classes = os.listdir(dataset_dir)
        for word_class in word_classes:
            class_dir = os.path.join(dataset_dir, word_class)
            if not os.path.isdir(class_dir):
                continue

            videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            for video in videos:
                video_path = os.path.join(class_dir, video)
                print(f'Processing {video_path}...')

                try:
                    # 비디오에서 특징 추출 (증강 적용)
                    features = extract_features_from_video(video_path, augment=augment, num_augments=num_augments)

                    if features:  # 특징이 비어 있지 않은 경우에만 저장
                        save_path = os.path.join(class_dir, f'{os.path.splitext(video)[0]}_features.npy')
                        np.save(save_path, features)
                        print(f'Features saved to {save_path}')
                    else:
                        print(f"No features extracted from {video_path}")
                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
    except Exception as e:
        print(f"Error processing dataset directory {dataset_dir}: {str(e)}")

if __name__ == "__main__": 
    dataset_path = r'/mnt/8TB_2/sohyun/sonic/sonic_ml/raw_dataset/words'
    process_dataset(dataset_path, augment=True, num_augments=5)
