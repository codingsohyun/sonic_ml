from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from finger_spelling.finger_stream import gen_frames  # 프레임 생성 및 유사도 반환
from real_time_inference import real_time_inference  # 실시간 추론을 위해 import

app = Flask(__name__)

# 홈 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 실시간 웹캠 스트리밍 피드
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 실시간 추론 결과를 반환하는 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    # 실시간으로 유사도를 반환하는 함수 호출
    similarity, predicted_class = real_time_inference()

    # JSON 형식으로 예측된 클래스와 유사도를 반환
    return jsonify({
        'predicted_class': predicted_class,
        'similarity': similarity
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
