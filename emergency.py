import cv2
import base64
from ultralytics import YOLO

def findEmergency(image_path, conf=0.6, iou=0.4, max_det=20):
    """
    Runs YOLO object detection on a given image and returns:
      - number of detections
      - list of labels
      - annotated image as base64 string
    """

    # Load trained model (hardcoded path for simplicity)
    model = YOLO(r"models\emergency_best.pt")

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        max_det=max_det
    )

    detection_count = 0
    labels = []
    annotated_frame = None

    for r in results:
        detection_count += len(r.boxes)
        labels.extend([model.names[int(box.cls)] for box in r.boxes])
        annotated_frame = r.plot()

    if annotated_frame is not None:
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        image_base64 = None

    return detection_count, labels, image_base64


if __name__ == "__main__":
    detection_count, labels, img_b64 = findEmergency(r"suv_amb.jpg")
    print("Detections:", detection_count)
    print("Labels:", labels)
    print("Image (base64):", img_b64[:100], "...")
