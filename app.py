from flask import Flask, render_template, Response
import cv2
from finger_recon import recognize_finger_spelling

app = Flask(__name__)

# 웹캠에서 실시간 영상을 받아오는 함수
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 손 관절 인식 후 지문자 예측
            result = recognize_finger_spelling(frame)
            
            # 화면에 결과 출력
            cv2.putText(frame, str(result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
