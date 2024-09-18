import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_features_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    features = {}

    # 손 특징 추출
    if hand_results.multi_hand_landmarks:
        hand_data = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                hand_data.append((landmark.x, landmark.y, landmark.z))
        features['hand'] = hand_data

    # 신체 특징 추출
    if pose_results.pose_landmarks:
        pose_data = []
        for landmark in pose_results.pose_landmarks.landmark:
            pose_data.append((landmark.x, landmark.y, landmark.z))
        features['pose'] = pose_data

    return features

if __name__ == "__main__":
    # 테스트용 코드: 웹캠에서 프레임 읽고 특징 추출
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 특징 추출
        features = extract_features_from_frame(frame)
        print(features)  # 특징 출력

        cv2.imshow('Webcam Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
