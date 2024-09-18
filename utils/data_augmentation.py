import cv2
import os
import numpy as np
import random

def augment_image(image):
    """
    주어진 이미지에 대해 다양한 증강 기법을 적용합니다.
    """
    # 1. 좌우 반전
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 2. 밝기 조정
    if random.random() > 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.5, 1.5)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. 회전
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))

    return image

def augment_video(video_path, output_dir, num_augments=5):
    """
    주어진 비디오에 대해 num_augments만큼 증강된 비디오를 생성합니다.
    """
    cap = cv2.VideoCapture(video_path)
    frame_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)

    cap.release()

    # 증강된 비디오 생성
    for i in range(num_augments):
        augmented_frames = [augment_image(frame) for frame in frame_list]

        # 출력 비디오 저장
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(output_dir, f'augmented_{i}.mp4')
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (frame_list[0].shape[1], frame_list[0].shape[0]))

        for frame in augmented_frames:
            out.write(frame)
        out.release()

def augment_dataset(dataset_dir, output_dir):
    """
    데이터셋에 있는 모든 비디오에 대해 데이터 증강을 적용합니다.
    """
    classes = os.listdir(dataset_dir)
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]

        for video in videos:
            video_path = os.path.join(class_dir, video)
            print(f"Augmenting {video_path}...")
            augment_video(video_path, output_class_dir)

if __name__ == "__main__":
    dataset_path = r'D:\sonic_ml\raw_dataset\words'
    output_path = r'D:\sonic_ml\augmented_dataset\words'
    augment_dataset(dataset_path, output_path)
