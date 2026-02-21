# Real-Time Object Recognition System Using MediaPipe and OpenCV

## 1. Introduction

This project demonstrates the creation of a real-time object recognition system using **MediaPipe** and **OpenCV**. The system detects and classifies a small set of common objects (such as a cup, phone, etc.) from a live video stream provided by the webcam. 

The project leverages **MediaPipe's object detection** and the **EfficientDet Lite0** pre-trained model for classification. **OpenCV** is used for video capture and displaying the video with bounding boxes and object labels.

## 2. Objectives

The primary objectives of this project are:
- **Detect and recognize objects** in real-time using a webcam feed.
- **Display the detected objects** with bounding boxes and labels on the screen.
- Use a **pre-trained model** to classify the detected objects with their appropriate labels.

## 3. System Overview

The system processes real-time video input from the webcam, applies object detection using the pre-trained **MediaPipe model**, and visualizes the results by displaying bounding boxes and object labels on the video stream. **OpenCV** is used to display the annotated video in real-time.

## 4. Technologies Used

- **Python**: The core development language.
- **MediaPipe**: A cross-platform framework for building pipelines for object detection and classification.
- **OpenCV**: A computer vision library used for video capture and displaying results.
- **Pretrained Model**: The **EfficientDet Lite0** model from MediaPipe is used for object detection. This model is optimized for real-time applications on resource-constrained devices, such as mobile phones and edge devices, balancing accuracy and speed.
- **TensorFlow Lite**: The EfficientDet Lite0 model uses TensorFlow Lite for efficient on-device inference, eliminating the need for external computation resources.
- **COCO Dataset**: The model is pre-trained on the COCO dataset, which contains 80 common objects for detection. More information can be found at [COCO Dataset](https://cocodataset.org/#home).

## 5. Key Components

1. **Object Detection with MediaPipe**:
   - MediaPipe's Object Detector is used to detect common objects in the video stream.
   - The detector is configured using the **efficientdet_lite0.tflite** TensorFlow Lite model, providing a balance of speed and accuracy for real-time applications.

2. **Preprocessing and Visualization**:
   - Video frames are pre-processed with noise reduction (Gaussian blur) before being fed into the object detection model.
   - Bounding boxes are drawn around detected objects, and labels with their respective confidence scores are displayed.

3. **Real-Time Processing**:
   - The system continuously captures frames from the webcam using OpenCV, processes them with the MediaPipe object detection model, and visualizes the results (bounding boxes and labels) in real-time.

4. **User Interface**:
   - The annotated video feed is displayed in an OpenCV window.
   - The user can exit the system by pressing the **'Esc'** key.

5. **Result**:
   - A video of live object detection is recorded and stored in the **Results** folder.
   - The results folder contains 2 images and 2 video recordings that showcase the object detection functionality.

## 6. Conclusion

This project successfully integrates **MediaPipe** and **OpenCV** to perform real-time object recognition and classification using a pre-trained model. The system is efficient and displays object detection results in real-time with bounding boxes and labels.

## 7. Future Improvements

- Integration of more advanced models for higher accuracy (e.g., **SSD** or **YOLO**).
- Adaptation to specific domains and fine-tuning of the model for specific contexts. The current model only detects common objects, so domain-specific adaptations will be necessary for particular use cases.

## 8. How to Run the Project

To run this project, follow the steps below:

1. Install the required libraries:

   ```bash
   pip install mediapipe opencv-python numpy tensorflow

2. Run the code using:
   ```bash
   python object_recognition.py
   
4. Monitor the model's object detection in the webcam feed.
5. Press 'Esc' to terminate the system.



  
