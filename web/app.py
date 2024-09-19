from flask import Flask, render_template, Response, jsonify, request
import threading
from streams.finger_stream import gen_frames  # 프레임 생성 및 유사도 반환
from models.finger.finger_recon import finger_inference 
from models.body.body_recon import body_inference 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def time_limit(result_event, response_data):
    # 15초 대기
    if not result_event.wait(timeout=15):
        response_data['result'] = 0

# 지문자 배우기 
@app.route('/finger_learn', methods=['POST'])
def finger_learn():
    data = request.json
    finger_id = data.get('id')
    # 확인..
    print(f"{finger_id}")

    # 손가락 inference에서 예측 클래스 가져오기
    predicted_class = finger_inference(finger_id)   

    # 클라이언트에서 넘어온 클래스_id와 예측 클래스_id가 맞으면 1, 틀렸으면 0이라고 연송이한테 넘겨주기  
    result = 1 if predicted_class == finger_id else 0
    # 확인..
    print(result)
    return jsonify({'result': result})

# 지문자 퀴즈 
@app.route('/finger_quiz', methods=['POST'])
def finger_quiz():
    data = request.json
    finger_id = data.get('id')
    # 확인..
    print(f"{finger_id}")

    response_data = {'result': None}
    result_event = threading.Event()
    thread = threading.Thread(target=time_limit, args=(result_event, response_data))
    thread.start()

    # 손가락 inference에서 예측 클래스 가져오기
    predicted_class = finger_inference(finger_id)   

    # 클라이언트에서 넘어온 클래스_id와 예측 클래스_id가 맞으면 1, 틀렸으면 0이라고 연송이한테 넘겨주기  
    response_data['result'] = 1 if predicted_class == finger_id else 0
    # 확인..
    print(f"{'results': response_data['result']}")

    result_event.set()

    return jsonify(response_data)

# 단어 배우기
@app.route('/body_learn', methods=['POST'])
def body_learn():
    data = request.json
    body_id = data.get('id')
    # 확인..
    print(f"{body_id}")

    # 단어 inference에서 예측 클래스 가져오기
    predicted_class = body_inference(body_id)

    # 맞았으면 1, 틀렸으면 0이라고 연송이한테 넘겨주기
    result = 1 if predicted_class == body_id else 0

    return jsonify({'result': result})

# 단어 퀴즈 
@app.route('/body_quiz', methods=['POST'])
def body_quiz():
    data = request.json
    body_id = data.get('id')
    # 확인..
    print(f"{body_id}")

    response_data = {'result': None}
    result_event = threading.Event()
    thread = threading.Thread(target=time_limit, args=(result_event, response_data))
    thread.start()

    # 손가락 inference에서 예측 클래스 가져오기
    predicted_class = body_inference(body_id)   

    # 클라이언트에서 넘어온 클래스_id와 예측 클래스_id가 맞으면 1, 틀렸으면 0이라고 연송이한테 넘겨주기  
    response_data['result'] = 1 if predicted_class == body_id else 0
    # 확인..
    print(f"{'results': response_data['result']}")

    result_event.set()

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
