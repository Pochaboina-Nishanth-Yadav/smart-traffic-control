import cv2
import base64
from ultralytics import YOLO


def detectHelmet(image_path, conf=0.4, iou=0.7, max_det=20):
    """
    Detect bikes and helmet violations.

    Returns:
      - detection_count
      - labels (final labels like 'bike | helmet', 'no_helmet')
      - annotated image (base64)
    """

    # Load model
    model = YOLO(r"models/helmet.pt")

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        max_det=max_det
    )

    detection_count = 0
    final_labels = []
    annotated_img = None

    for r in results:
        img = r.orig_img.copy()

        bike_boxes = []
        helmet_boxes = []
        nohelmet_boxes = []

        # Separate detections
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            coords = list(map(int, box.xyxy[0]))

            if label == "bike":
                bike_boxes.append(coords)
            elif label == "helmet":
                helmet_boxes.append(coords)
            elif label == "no_helmet":
                nohelmet_boxes.append(coords)

        # ✅ Process bikes
        for (x1, y1, x2, y2) in bike_boxes:
            bike_label = "bike"

            # Check no_helmet inside bike
            for (ox1, oy1, ox2, oy2) in nohelmet_boxes:
                if ox1 >= x1 and oy1 >= y1 and ox2 <= x2 and oy2 <= y2:
                    bike_label = "bike | no_helmet"
                    break

            # Check helmet if no violation
            if bike_label == "bike":
                for (ox1, oy1, ox2, oy2) in helmet_boxes:
                    if ox1 >= x1 and oy1 >= y1 and ox2 <= x2 and oy2 <= y2:
                        bike_label = "bike | helmet"

            # Draw
            color = (0, 0, 255) if "no_helmet" in bike_label else (0, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, bike_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            final_labels.append(bike_label)
            detection_count += 1

        # ✅ If NO BIKE → show no_helmet
        if len(bike_boxes) == 0:
            for (x1, y1, x2, y2) in nohelmet_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "no_helmet", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                final_labels.append("no_helmet")
                detection_count += 1

        annotated_img = img

    # Convert to base64
    if annotated_img is not None:
        _, buffer = cv2.imencode('.jpg', annotated_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        image_base64 = None

    return detection_count, final_labels, image_base64