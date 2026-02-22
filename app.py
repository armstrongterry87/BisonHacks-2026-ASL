import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --- Page Layout ---
st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("ASL Real-Time Translator")

# Create two columns: Left for Video, Right for big Text
col1, col2 = st.columns([2, 1])

with col1:
    frame_placeholder = st.empty() # Video goes here

with col2:
    st.subheader("Predicted Character")
    # This big HTML block makes the letter look huge and clear
    result_text = st.empty() 

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Process the image
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    predicted_char = "..." # This will be your model's output later

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # DRAWING METHOD 1: Draw the hand skeleton connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # DRAWING METHOD 2: Text Overlay on the video
            # We put the text near the "Wrist" landmark (index 0)
            x = int(hand_landmarks.landmark[0].x * frame.shape[1])
            y = int(hand_landmarks.landmark[0].y * frame.shape[0])
            
            cv2.putText(frame, predicted_char, (x, y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # 2. Update the Screen (Frontend)
    # Update the video feed
    frame_placeholder.image(frame, channels="BGR")
    
    # Update the big sidebar text
    result_text.markdown(f"""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="font-size: 100px; color: #FF4B4B;">{predicted_char}</h1>
        </div>
    """, unsafe_allow_html=True)

cap.release()
