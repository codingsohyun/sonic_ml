import cv2

def stream_webcam():
    cap = cv2.VideoCapture(0)  # 기본 웹캠 사용 (ID: 0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # 웹캠 화면 표시
        cv2.imshow('Webcam Stream', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_webcam()
