# from countVehicle import countVehicles
# c,g=countVehicles("traffic.mp4")
# print(c,g)

from ultralytics import YOLO
import cv2

# Load model
model = YOLO(r"C:\\Users\\NISHANTH\\Downloads\\Major\\backend\\models\\helmet.pt")

image_path = r"C:\\Users\\NISHANTH\\Downloads\\Major\\backend\\test\\no_helmet2.jpg"

results = model.predict(source=image_path, conf=0.7, iou=0.7)

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

        # Check inside bike
        for (ox1, oy1, ox2, oy2) in nohelmet_boxes:
            if ox1 >= x1 and oy1 >= y1 and ox2 <= x2 and oy2 <= y2:
                bike_label = "bike | no_helmet"
                break

        if bike_label == "bike":
            for (ox1, oy1, ox2, oy2) in helmet_boxes:
                if ox1 >= x1 and oy1 >= y1 and ox2 <= x2 and oy2 <= y2:
                    bike_label = "bike | helmet"

        color = (0, 0, 255) if "no_helmet" in bike_label else (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, bike_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Handle NO BIKE case (standalone no_helmet)
    if len(bike_boxes) == 0:
        for (x1, y1, x2, y2) in nohelmet_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "no_helmet", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Save output
cv2.imwrite("output.jpg", img)

print("Done! Check output.jpg")