# Real-Time ASL Recognition & Accessible CAPTCHA System

## 1. Introduction

This project demonstrates the creation of a real-time American Sign Language (ASL) recognition system using **MediaPipe** and **Scikit-Learn**. Initially designed as an ASL translator, the system also functions as an inclusive, biometric multi-factor authentication (MFA) tool. 

By asking users to perform specific ASL signs rather than clicking on blurry images of traffic lights, this project provides a "Next-Generation CAPTCHA" that enhances fairness and accessibility for Deaf and hard-of-hearing users during high-traffic e-commerce checkout queues.

## 2. Objectives

The primary objectives of this project are:
- **Track hand movements** in real-time by extracting 3D spatial coordinates using a live webcam feed.
- **Classify ASL signs** instantly using a custom-trained, lightweight machine learning model.
- **Provide an accessible security alternative** to traditional image-based CAPTCHAs.
- **Ensure user privacy** by processing mathematical coordinate data entirely locally, ensuring no video or images are ever transmitted or saved.

## 3. System Overview

The system processes real-time video input from the user's webcam. Instead of analyzing raw image pixels, it uses **MediaPipe Hands** to extract the exact `(x, y, z)` coordinates of 21 landmarks on the user's hand. These coordinates are fed into a **Scikit-Learn Random Forest Classifier**, which predicts the corresponding ASL letter. **OpenCV** is used to display a dynamic user interface, rendering a tracking skeleton, live confidence percentages, and a translation text bar.

## 4. Technologies Used

* **Python**: The core development language.
* **MediaPipe**: A cross-platform framework used for real-time computer vision. In this project, its hands solution is utilized to track and extract the precise 3D spatial coordinates of 21 hand landmarks from the live video feed.
* **OpenCV**: A computer vision library used for video capture, rendering the live webcam stream, and drawing the visual interface elements (such as the translation text bar, tracking skeleton, and confidence percentages) directly on the screen.
* **Pandas**: A data analysis and manipulation library utilized during the model training phase. It was essential for loading, parsing, and structuring the custom CSV dataset of hand landmarks, allowing the seamless separation of coordinate data from the ASL letter labels.
* **Scikit-Learn**: A machine learning library used to build the core predictive "brain" of the application. We implemented a Random Forest Classifier to analyze the spatial hand coordinates extracted by MediaPipe and accurately translate them into the correct ASL signs in real time.

## 5. Key Components

1. **Hand Landmark Extraction**:
   - MediaPipe Hands isolates the user's hand from the background and maps 21 specific joints, converting visual data into a lightweight array of 63 numbers `(x, y, z)`.
   
2. **Custom Machine Learning Model**:
   - The predictive model is a Random Forest Classifier trained on a custom dataset (`asl_landmarks.csv`). Because it trains on structured coordinate data rather than heavy image files, the model is highly accurate, extremely fast, and computationally inexpensive.

3. **Debounce & Verification Logic**:
   - To prevent erratic spamming of letters, the system includes a "lock-in" verification mechanism. The model must recognize a sign with at least 70% confidence for 15 consecutive frames (~0.5 seconds) before it registers the input as a success.

4. **User Interface**:
   - The annotated video feed provides live visual feedback, changing text colors from yellow to green when high-confidence predictions are achieved.
   - The interface acts as a seamless security or translation window directly overlaying the live feed.

## 6. Conclusion

This project successfully integrates MediaPipe, Scikit-Learn, and OpenCV to perform real-time ASL recognition. By combining lightweight machine learning with privacy-by-design architecture (local coordinate processing), the system serves as both a functional communication tool and a proof-of-concept for accessible, human-first application security and bot-mitigation.

## 7. Future Improvements

- **Dynamic Gesture Recognition**: Expanding the model to recognize moving ASL phrases rather than static alphabet letters.
- **Web Integration**: Porting the Python backend into a web framework (like Flask or FastAPI) to seamlessly integrate the ASL CAPTCHA step directly into a live e-commerce checkout flow.
- **Adaptive Difficulty**: Implementing dynamic step-up authentication that requires longer or more complex sign sequences during severe bot-attacks or high-traffic queue surges.

## 8. How to Run the Project

To run this project locally, follow the steps below:

1. Install the required dependencies:
   ```bash
   pip install mediapipe opencv-python pandas scikit-learn numpy
2. Ensure your trained model (asl_model.pkl) is in the same directory as your script
3. Run the live application using: python app.py
4. Monitor the model's ASL detection in the webcam feed. Hold an ASL sign confidently to type it on the screen.
5. Press the 'c' key to clear the typed text.
6. Press the 'q' key to close the window and terminate the system.
