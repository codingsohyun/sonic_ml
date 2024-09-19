import cv2
import mediapipe as mp
from models.body.body_inference import body_inference

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

drawing_spec_landmarks = mp_drawing.DrawingSpec(color=(196, 148, 130), thickness=3, circle_radius=3)
drawing_spec_connections = mp_drawing.DrawingSpec(color=(214, 177, 172), thickness=3)

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

            similarity = body_inference(frame)
            
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
