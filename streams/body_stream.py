import sys
import os
sys.path.append('D:/sonic_ml')

import cv2
import mediapipe as mp
from models.body.body_inference import body_inference

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

drawing_spec_landmarks = mp_drawing.DrawingSpec(color=(196, 148, 130), thickness=3, circle_radius=3)
drawing_spec_connections = mp_drawing.DrawingSpec(color=(214, 177, 172), thickness=3)

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

def body_stream():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # 몸의 관절이 감지되면 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    drawing_spec_landmarks,
                    drawing_spec_connections
                )

            # 특징 추출
            features = extract_features_from_frame(frame)

            # 모델을 통해 유사도 계산
            similarity, predicted_class = body_inference(frame)

            if similarity == 100:
                display_text = "Match"
            else:
                display_text = f"{similarity}%"

            # 유사도를 화면의 왼쪽 중앙에 출력
            height, width, _ = frame.shape
            y_position = height // 2  # 화면의 수직 중앙
            cv2.putText(frame, display_text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (196, 148, 130), 2, cv2.LINE_AA)
            
            # 프레임을 JPEG로 인코딩 후 스트리밍
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
