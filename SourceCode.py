import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
import joblib

# Constants for visualization
MARGIN = 10  
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0) 

session_history = []

def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and returns it."""

    count = len(detection_result.detections)

    
    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        
        # Dynamic color based on detection confidence
        probability = round(detection.categories[0].score, 2)
        
        if probability > 0.8:
            TEXT_COLOR = (0, 255, 0)  # Green for high confidence
        else:
            TEXT_COLOR = (0, 255, 255)  # Yellow for low confidence
        
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category_name = detection.categories[0].category_name
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + int(bbox.origin_x), MARGIN + ROW_SIZE + int(bbox.origin_y))
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image, count

# STEP 1: Create an ObjectDetector object with more accurate model
base_options = python.BaseOptions(model_asset_path='./pretrainedModel/efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.4)  
detector = vision.ObjectDetector.create_from_options(options)

# STEP 2: Initialize the video capture with higher resolution for better accuracy
cap = cv2.VideoCapture(0)

# Set resolution for higher accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Real-time video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame (optional noise reduction and histogram equalization)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.GaussianBlur(image_rgb, (5, 5), 0)  # Noise reduction

    # Create a MediaPipe Image from the numpy array with the correct image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect objects
    detection_result = detector.detect(mp_image)

    # Visualize the bounding boxes and labels
    annotated_image, current_obj_count = visualize(frame, detection_result)

    # Display the annotated image in a window
    cv2.imshow('Real-Time Object Detection', annotated_image)

    # Exit the loop when the user presses the 'Esc' key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# --- NEW: POST-PROCESS AND TRAIN ---
if session_history:
    # 1. Convert to Pandas DataFrame
    df = pd.DataFrame(session_history)
    print("\nSession Data Summary:")
    print(df.describe())

    # 2. Simple Scikit-Learn Training
    # We'll train a model to predict if a scene is "Active" (count > 0)
    X = df[['objects_detected']]
    y = (df['objects_detected'] > 0).astype(int)
    
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    
    # 3. Save the model for future use
    joblib.dump(clf, 'mvp_model.pkl')
    print("\nSuccess: session data saved and model 'mvp_model.pkl' trained.")

