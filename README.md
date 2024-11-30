Object Detection Web App with YOLO and Flask
===========================================

This project is a web application that performs real-time object detection using the YOLO (You Only Look Once) model, integrated with Flask and OpenCV. The app allows users to upload an image or use their webcam for detecting various objects in real-time.

Requirements
------------
1. Python 3.x
2. Flask
3. OpenCV (cv2)
4. YOLOv3 weights, configuration file, and class names file
5. A modern web browser (Chrome, Firefox, etc.)

Installation
------------
1. Clone the repository:

2. Install the required Python dependencies:

3. Download the following YOLOv3 files:
- yolov3.weights
- yolov3.cfg
- coco.names
Place these files in the `yolo_web_app` directory.

4. Run the Flask app:

5. Open a web browser and navigate to:

Features
--------
1. **Image Upload for Object Detection**: Allows users to upload an image file for object detection. The detected objects will be highlighted with bounding boxes and labeled with their names and confidence scores.

2. **Webcam Real-Time Detection**: Users can start the webcam and perform real-time object detection. Detected objects will be continuously updated on the screen.

3. **Dark Mode Toggle**: The app supports both light and dark modes. Users can toggle between the modes for a better viewing experience.

4. **Detected Items**: The app displays a list of detected objects with their confidence scores after each detection.

How to Use
-----------
1. Navigate to the app's main page.
2. Upload an image using the file input or start the webcam detection by clicking on the "Start Webcam Detection" button.
3. The app will display the uploaded or detected image with bounding boxes around the detected objects. It will also list the objects detected with their confidence scores.

File Structure
--------------
- `yolo_web_app/`
- `app.py`: The main Flask application file.
- `yolov3.weights`: The pre-trained YOLOv3 weights file.
- `yolov3.cfg`: The YOLOv3 configuration file.
- `coco.names`: The file containing the class labels for YOLO.
- `templates/`
 - `index.html`: The HTML template for the main page of the web app.
 - `webcam.html`: A dedicated page for webcam object detection.
- `requirements.txt`: A file listing Python dependencies.

Troubleshooting
---------------
- **No webcam detection**: Ensure your browser has permissions to access the webcam.
- **Error loading YOLO files**: Verify that the `yolov3.weights`, `yolov3.cfg`, and `coco.names` files are located in the `yolo_web_app` directory.

Contributing
------------
Feel free to fork this repository and contribute by submitting pull requests or creating issues if you encounter bugs or have feature suggestions.

License
-------
This project is licensed under the MIT License.

Author
------
Syed Abdul Qadeer



You will need to download the weights file (yolov3.weights) and place it inside the same folder
from https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights
