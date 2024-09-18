import cv2
import mediapipe as mp
from finger_spelling.finger_recon import recognize_finger_spelling  # 이 함수는 유사도를 반환해야 함

# MediaPipe 손 인식 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# 사용자 지정 색상 및 그리기 사양 설정
drawing_spec_landmarks = mp_drawing.DrawingSpec(color=(196, 148, 130), thickness=3, circle_radius=3)
drawing_spec_connections = mp_drawing.DrawingSpec(color=(214, 177, 172), thickness=3)

# 웹캠에서 실시간 영상을 받아오는 함수
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # BGR에서 RGB로 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # 손 관절이 감지되면 그리기
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS, 
                        drawing_spec_landmarks, 
                        drawing_spec_connections
                    )

            # 손 관절 인식 후 지문자 예측 및 유사도 측정
            similarity = recognize_finger_spelling(frame)
            
            # 유사도에 따라 화면에 결과 출력
            if similarity == 100:
                display_text = "Match"
                print(1)  # 유사도가 100일 때 1 출력
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