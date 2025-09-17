from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64
import os

# Load YOLO model and class labels
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels from a file (e.g., coco.names)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

    

app = Flask(__name__)

def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.2)

    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []

    if len(indices) > 0:
        for i in indices.flatten():
            filtered_boxes.append(boxes[i])
            filtered_confidences.append(confidences[i])
            filtered_class_ids.append(class_ids[i])

    return filtered_boxes, filtered_class_ids, filtered_confidences

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    image_data = data.get("image")

    # Decode the base64 image
    img_data = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detect objects
    boxes, class_ids, confidences = detect_objects(image)

    # Annotate image with detections
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]]) if class_ids[i] < len(classes) else "Unknown"
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode image back to bytes
    _, encoded_img = cv2.imencode(".jpg", image)
    img_data = encoded_img.tobytes()

    # Convert to base64
    img_str = base64.b64encode(img_data).decode('utf-8')

    detected_items = [{"name": classes[class_ids[i]], "confidence": confidences[i]} for i in range(len(class_ids))]

    return jsonify({
        "detected_items": detected_items,
        "image": img_str
    })

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Read the image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Detect objects
        boxes, class_ids, confidences = detect_objects(image)

        # Annotate image with detections
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) if class_ids[i] < len(classes) else "Unknown"
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode image back to bytes
        _, encoded_img = cv2.imencode(".jpg", image)
        img_data = encoded_img.tobytes()

        # Convert to base64
        img_str = base64.b64encode(img_data).decode('utf-8')

        detected_items = [{"name": classes[class_ids[i]], "confidence": confidences[i]} for i in range(len(class_ids))]

        return jsonify({
            "detected_items": detected_items,
            "image": img_str
        })

if __name__ == "__main__":
    app.run(debug=True)
