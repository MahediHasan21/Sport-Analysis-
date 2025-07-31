import cv2

video_path = "/Users/apple/FINAL PROJECT AND REPORTS/EGA&TFP/input_videos/08fd33_4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: OpenCV cannot open the video file. Check the file path and format.")
else:
    print("✅ OpenCV successfully opened the video file.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

cap.release()

print(f"📸 Extracted {frame_count} frames from the video.")


