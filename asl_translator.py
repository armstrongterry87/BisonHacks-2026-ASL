import cv2
import mediapipe as mp
import pickle
import time
import numpy as np

# Load the trained model
with open('asl_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Setup MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize the model options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7)

landmarker = HandLandmarker.create_from_options(options)

# Debounce settings
FRAMES_TO_LOCK = 15  # Number of frames to hold a sign before it "types"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider a prediction

# Tracking variables
current_letter = None
letter_frame_count = 0
letter_typed = False  # NEW: Flag to prevent repeated typing
typed_text = ""

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()

print("ASL Translator Running!")
print("Press 'c' to clear text, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp
    timestamp_ms = int((time.time() - start_time) * 1000)
    
    # Process the frame
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    predicted_letter = ""
    confidence = 0.0
    
    # If a hand is detected
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        
        # Draw green circles on the joints
        for lm in hand:
            x_px, y_px = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
        
        # Flatten the landmarks into a single list for prediction
        landmark_row = []
        for lm in hand:
            landmark_row.extend([lm.x, lm.y, lm.z])
        
        # Make prediction
        landmark_array = np.array(landmark_row).reshape(1, -1)
        prediction = model.predict(landmark_array)
        probabilities = model.predict_proba(landmark_array)
        
        predicted_letter = prediction[0]
        confidence = max(probabilities[0])
        
        # Debounce logic
        if confidence >= CONFIDENCE_THRESHOLD:
            if predicted_letter == current_letter:
                letter_frame_count += 1
                # Only type once when threshold is reached
                if letter_frame_count == FRAMES_TO_LOCK and not letter_typed:
                    typed_text += predicted_letter
                    letter_typed = True  # Mark as typed
                    print(f"Typed: {predicted_letter}")
            else:
                # New letter detected, reset counter and flag
                current_letter = predicted_letter
                letter_frame_count = 1
                letter_typed = False  # Reset flag for new letter
        else:
            # Confidence too low, reset
            current_letter = None
            letter_frame_count = 0
            letter_typed = False
    else:
        # No hand detected, reset
        current_letter = None
        letter_frame_count = 0
        letter_typed = False
    
    # Calculate progress bar for debounce
    progress = min(letter_frame_count / FRAMES_TO_LOCK, 1.0)
    
    # Display prediction and confidence in top left
    if predicted_letter:
        cv2.putText(frame, f"Sign: {predicted_letter}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw progress bar for debounce
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (10, 90), (210, 110), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 90), (210, 110), (255, 255, 255), 2)
        
        # Show "TYPED!" indicator when letter is locked
        if letter_typed:
            cv2.putText(frame, "TYPED!", (220, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Create text display area at the bottom
    text_area_height = 80
    text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)
    
    # Display typed text
    cv2.putText(text_area, "Typed Text:", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    cv2.putText(text_area, typed_text[-30:] if len(typed_text) > 30 else typed_text, 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Combine frame and text area
    combined_frame = np.vstack([frame, text_area])
    
    # Display instructions
    cv2.putText(combined_frame, "Press 'c' to clear | 'q' to quit", 
                (combined_frame.shape[1] - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow('ASL Translator', combined_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        typed_text = ""
        print("Text cleared!")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal typed text: {typed_text}")
print("ASL Translator closed.")