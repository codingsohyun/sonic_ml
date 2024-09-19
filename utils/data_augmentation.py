import cv2
import random

def augment_image(image):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    if random.random() > 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.5, 1.5)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))

    return image

def augment_video_frames(frame_list, num_augments=5):
    augmented_videos = []
    
    for _ in range(num_augments):
        augmented_frames = [augment_image(frame) for frame in frame_list]
        augmented_videos.append(augmented_frames)
    
    return augmented_videos
