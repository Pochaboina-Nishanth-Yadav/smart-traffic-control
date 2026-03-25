from flask import Flask, request, jsonify
from flask_cors import CORS
from helmet import detectHelmet
import os
from countVehicle import process_video
from emergency import findEmergency
from wrong import detectWrongSide

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    video_path = os.path.join("uploads", file.filename)
    file.save(video_path)

    vehicle_count, green_time, image_base64 = process_video(video_path)

    if vehicle_count is None:
        return jsonify({"error": "Failed to read video"}), 500

    return jsonify({
        "vehicle_count": vehicle_count,
        "green_time": green_time,
        "image": f"data:image/jpeg;base64,{image_base64}"
    })


@app.route('/emergency', methods=['POST'])
def emergency():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    detection_count, labels, image_base64 = findEmergency(image_path)

    if detection_count is None:
        return jsonify({"error": "Failed to process image"}), 500

    return jsonify({
        "detection_count": detection_count,
        "labels": labels,
        "image": f"data:image/jpeg;base64,{image_base64}"
    })
@app.route('/detect-helmet', methods=['POST'])
def detect_helmet_api():
    try:
        # 1. Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # 2. Save temporarily
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # 3. Call your function
        count, labels, image_base64 = detectHelmet(file_path)

        # 4. Delete file (optional cleanup)
        os.remove(file_path)

        # 5. Return response
        return jsonify({
            "detection_count": count,
            "labels": labels,
            "image": f"data:image/jpeg;base64,{image_base64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect-wrong-side', methods=['POST'])
def detect_wrong_side_api():
    try:
        # 1. Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # 2. Save file
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # 3. Call detection
        count, labels, image_base64 = detectWrongSide(file_path)

        # 4. Delete file
        os.remove(file_path)

        # 5. Return response
        return jsonify({
            "detection_count": count,
            "labels": labels,
            "image": f"data:image/jpeg;base64,{image_base64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
