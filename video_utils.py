import cv2
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames, output_path, fps=30):
    if not frames or not isinstance(frames, list):
        print("❌ ERROR: Frames are empty or not a list.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use a codec like 'XVID' or 'MJPG'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            print(f"❌ ERROR: Frame {i} is not a valid numpy array.")
            continue
        out.write(frame)

    out.release()
    print(f"✅ Video saved to {output_path}")




