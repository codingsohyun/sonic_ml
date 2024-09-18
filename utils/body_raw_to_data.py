import cv2
import mediapipe as mp
import os

# MediaPipe 사용을 위한 설정
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 손 및 신체 특징 추출 함수
def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 손, 신체 특징 추출
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        frame_features = {}

        # 손 특징 추출
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append((landmark.x, landmark.y, landmark.z))
                frame_features['hand'] = hand_data

        # 신체 특징 추출
        if pose_results.pose_landmarks:
            pose_data = []
            for landmark in pose_results.pose_landmarks.landmark:
                pose_data.append((landmark.x, landmark.y, landmark.z))
            frame_features['pose'] = pose_data

        if frame_features:
            features.append(frame_features)

    cap.release()
    return features

# 데이터셋 폴더 탐색 및 특징 추출
def process_dataset(dataset_dir):
    word_classes = os.listdir(dataset_dir)
    for word_class in word_classes:
        class_dir = os.path.join(dataset_dir, word_class)
        if not os.path.isdir(class_dir):
            continue

        videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        for video in videos:
            video_path = os.path.join(class_dir, video)
            print(f'Processing {video_path}...')
            
            # 비디오에서 특징 추출
            features = extract_features_from_video(video_path)
            
            # 특징 저장 (예: numpy 배열로 변환하여 .npy 파일로 저장 가능)
            save_path = os.path.join(class_dir, f'{os.path.splitext(video)[0]}_features.npy')
            np.save(save_path, features)
            print(f'Features saved to {save_path}')

# 메인 실행 부분
if __name__ == "__main__":
    dataset_path = r'D:/sonic_ml/raw_dataset/words'
    process_dataset(dataset_path)

