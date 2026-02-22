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
    min_hand_detection_confidence=0.5)  # Lowered for better detection

landmarker = HandLandmarker.create_from_options(options)

# Debounce settings
FRAMES_TO_LOCK = 20  # Increased for more stability
CONFIDENCE_THRESHOLD = 0.4  # Lowered threshold
SMOOTHING_FRAMES = 5  # Number of frames to average predictions

# Tracking variables
current_letter = None
letter_frame_count = 0
letter_typed = False
typed_text = ""
recent_predictions = []  # For smoothing

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()

print("ASL Translator Running!")
print("Press 'c' to clear text, 'q' to quit, 'b' for backspace")

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist and hand scale - MUST MATCH TRAINING"""
    lm_array = np.array(landmarks).reshape(21, 3)
    
    # Center around wrist (landmark 0)
    wrist = lm_array[0]
    centered = lm_array - wrist
    
    # Scale by hand size (distance to middle finger tip)
    middle_tip = centered[12]
    scale = np.linalg.norm(middle_tip)
    
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    return normalized.flatten()

def get_smoothed_prediction(recent_preds):
    """Get the most common prediction from recent frames"""
    if not recent_preds:
        return None, 0.0
    
    # Count occurrences
    from collections import Counter
    counts = Counter([p[0] for p in recent_preds])
    most_common = counts.most_common(1)[0]
    
    # Get average confidence for that letter
    letter = most_common[0]
    confidences = [p[1] for p in recent_preds if p[0] == letter]
    avg_confidence = sum(confidences) / len(confidences)
    
    return letter, avg_confidence

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
    top_predictions = []
    
    # If a hand is detected
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        
        # Draw green circles on the joints
        for lm in hand:
            x_px, y_px = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
        
        # Draw connections between landmarks for better visualization
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        for start, end in connections:
            start_point = (int(hand[start].x * frame.shape[1]), int(hand[start].y * frame.shape[0]))
            end_point = (int(hand[end].x * frame.shape[1]), int(hand[end].y * frame.shape[0]))
            cv2.line(frame, start_point, end_point, (0, 200, 0), 2)
        
        # Flatten the landmarks into a single list
        landmark_row = []
        for lm in hand:
            landmark_row.extend([lm.x, lm.y, lm.z])
        
        # IMPORTANT: Apply same normalization as training!
        normalized = normalize_landmarks(landmark_row)
        landmark_array = normalized.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(landmark_array)
        probabilities = model.predict_proba(landmark_array)
        
        raw_letter = prediction[0]
        raw_confidence = max(probabilities[0])
        
        # Add to recent predictions for smoothing
        recent_predictions.append((raw_letter, raw_confidence))
        if len(recent_predictions) > SMOOTHING_FRAMES:
            recent_predictions.pop(0)
        
        # Get smoothed prediction
        predicted_letter, confidence = get_smoothed_prediction(recent_predictions)
        
        # Get top 3 predictions for debugging
        top_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_predictions = [(model.classes_[i], probabilities[0][i]) for i in top_indices]
        
        # Debounce logic
        if confidence >= CONFIDENCE_THRESHOLD:
            if predicted_letter == current_letter:
                letter_frame_count += 1
                if letter_frame_count == FRAMES_TO_LOCK and not letter_typed:
                    typed_text += predicted_letter
                    letter_typed = True
                    print(f"Typed: {predicted_letter}")
            else:
                current_letter = predicted_letter
                letter_frame_count = 1
                letter_typed = False
        else:
            current_letter = None
            letter_frame_count = 0
            letter_typed = False
    else:
        current_letter = None
        letter_frame_count = 0
        letter_typed = False
        recent_predictions = []  # Clear predictions when no hand
    
    # Calculate progress bar for debounce
    progress = min(letter_frame_count / FRAMES_TO_LOCK, 1.0)
    
    # Display prediction and confidence in top left
    if predicted_letter:
        # Color based on confidence
        if confidence >= 0.7:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence >= 0.5:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 165, 255)  # Orange - low confidence
        
        cv2.putText(frame, f"Sign: {predicted_letter}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show top 3 predictions
        y_pos = 200
        cv2.putText(frame, "Top predictions:", (10, y_pos - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for letter, prob in top_predictions:
            bar_len = int(prob * 150)
            cv2.rectangle(frame, (10, y_pos), (10 + bar_len, y_pos + 15), (255, 200, 0), -1)
            cv2.putText(frame, f"{letter}: {prob*100:.1f}%", (170, y_pos + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
        
        # Draw progress bar for debounce
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (10, 90), (210, 110), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 90), (210, 110), (255, 255, 255), 2)
        
        if letter_typed:
            cv2.putText(frame, "TYPED!", (220, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Show your hand clearly", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
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
    cv2.putText(combined_frame, "'c' clear | 'b' backspace | 'q' quit", 
                (combined_frame.shape[1] - 320, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('ASL Translator', combined_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        typed_text = ""
        print("Text cleared!")
    elif key == ord('b'):
        typed_text = typed_text[:-1]
        print("Backspace!")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal typed text: {typed_text}")
print("ASL Translator closed.")