import cv2
import base64
from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        cap.release()
        return None, None, None

    # Resize frame for faster inference
    frame = cv2.resize(frame, (320, 320))

    # Run YOLO detection (only vehicle classes: car, motorcycle, bus, truck)
    results = model(frame, imgsz=320, classes=[2, 3, 5, 7])

    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            vehicle_count += 1

            # Get box coordinates safely
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            cv2.putText(frame, "Vehicle", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Example logic: green time proportional to vehicle count
    green_time = vehicle_count * 2

    cap.release()

    # Convert annotated frame → base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return vehicle_count, green_time, image_base64
