import cv2
import base64
from ultralytics import YOLO


def detectWrongSide(image_path, conf=0.6, iou=0.4, max_det=20):
    """
    Detect wrong-side driving.

    Returns:
      - detection_count
      - labels
      - annotated image (base64)
    """

    # Load model
    model = YOLO(r"models/wrong_side.pt")

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        max_det=max_det
    )

    detection_count = 0
    labels = []
    annotated_img = None

    for r in results:
        detection_count += len(r.boxes)
        labels.extend([model.names[int(box.cls)] for box in r.boxes])

        annotated_img = r.plot()

    # Convert image to base64
    if annotated_img is not None:
        _, buffer = cv2.imencode('.jpg', annotated_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        image_base64 = None

    return detection_count, labels, image_base64