from flask import Flask, render_template, Response, jsonify, redirect, url_for
import cv2
import mediapipe as mp
import pickle
import time
import numpy as np
import random
import secrets
import re
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load the trained model
with open('asl_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5)

landmarker = HandLandmarker.create_from_options(options)

# ==================== SHARED UTILITIES ====================

def normalize_landmarks(landmarks):
    lm_array = np.array(landmarks).reshape(21, 3)
    wrist = lm_array[0]
    centered = lm_array - wrist
    scale = np.linalg.norm(centered[12])
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    return normalized.flatten()

def get_smoothed_prediction(recent_preds):
    if not recent_preds:
        return None, 0.0
    from collections import Counter
    counts = Counter([p[0] for p in recent_preds])
    most_common = counts.most_common(1)[0]
    letter = most_common[0]
    confidences = [p[1] for p in recent_preds if p[0] == letter]
    avg_confidence = sum(confidences) / len(confidences)
    return letter, avg_confidence

def draw_hand_skeleton(frame, hand):
    for lm in hand:
        x_px, y_px = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        cv2.circle(frame, (x_px, y_px), 6, (0, 255, 0), -1)
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    for s, e in connections:
        sp = (int(hand[s].x * frame.shape[1]), int(hand[s].y * frame.shape[0]))
        ep = (int(hand[e].x * frame.shape[1]), int(hand[e].y * frame.shape[0]))
        cv2.line(frame, sp, ep, (0, 200, 0), 2)

CORRECTIONS = {
    'teh': 'the', 'adn': 'and', 'taht': 'that', 'wiht': 'with',
    'hte': 'the', 'dont': "don't", 'cant': "can't", 'im': "I'm",
}

def simple_correct(text):
    if not text:
        return text
    words = text.split()
    corrected_words = [CORRECTIONS.get(w.lower(), w) for w in words]
    text = ' '.join(corrected_words)
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    text = re.sub(r'\bi\b', 'I', text)
    if text and text[-1] not in '.!?':
        text += '.'
    return text

# ==================== TRANSLATOR MODE ====================

translator_state = {
    'typed_text': '',
    'current_letter': None,
    'letter_frame_count': 0,
    'letter_typed': False,
    'recent_predictions': [],
    'start_time': time.time()
}

TRANSLATOR_FRAMES_TO_LOCK = 20
TRANSLATOR_CONFIDENCE = 0.4
TRANSLATOR_SMOOTHING = 5

def generate_translator_frames():
    global translator_state
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - translator_state['start_time']) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        predicted_letter = ""
        confidence = 0.0
        
        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            draw_hand_skeleton(frame, hand)
            
            landmark_row = []
            for lm in hand:
                landmark_row.extend([lm.x, lm.y, lm.z])
            
            normalized = normalize_landmarks(landmark_row)
            landmark_array = normalized.reshape(1, -1)
            
            prediction = model.predict(landmark_array)
            probabilities = model.predict_proba(landmark_array)
            
            raw_letter = prediction[0]
            raw_confidence = max(probabilities[0])
            
            translator_state['recent_predictions'].append((raw_letter, raw_confidence))
            if len(translator_state['recent_predictions']) > TRANSLATOR_SMOOTHING:
                translator_state['recent_predictions'].pop(0)
            
            predicted_letter, confidence = get_smoothed_prediction(translator_state['recent_predictions'])
            
            if confidence >= TRANSLATOR_CONFIDENCE:
                if predicted_letter == translator_state['current_letter']:
                    translator_state['letter_frame_count'] += 1
                    if translator_state['letter_frame_count'] == TRANSLATOR_FRAMES_TO_LOCK and not translator_state['letter_typed']:
                        translator_state['typed_text'] += predicted_letter
                        translator_state['letter_typed'] = True
                else:
                    translator_state['current_letter'] = predicted_letter
                    translator_state['letter_frame_count'] = 1
                    translator_state['letter_typed'] = False
            else:
                translator_state['current_letter'] = None
                translator_state['letter_frame_count'] = 0
                translator_state['letter_typed'] = False
            
            color = (0, 255, 0) if confidence >= 0.7 else (0, 255, 255) if confidence >= 0.5 else (0, 165, 255)
            cv2.putText(frame, f"Sign: {predicted_letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            progress = min(translator_state['letter_frame_count'] / TRANSLATOR_FRAMES_TO_LOCK, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (10, 90), (210, 110), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
            
            if translator_state['letter_typed']:
                cv2.putText(frame, "TYPED!", (220, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Show your hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            translator_state['recent_predictions'] = []
        
        cv2.putText(frame, "TRANSLATOR MODE", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 204, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==================== CAPTCHA MODE ====================

CAPTCHA_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
CAPTCHA_FRAMES_TO_VERIFY = 20  # Frames needed to verify each letter
CAPTCHA_CONFIDENCE = 0.45
CAPTCHA_TIMEOUT = 60  # 1 minute total
CAPTCHA_NUM_LETTERS = 3  # Number of letters to sign

captcha_state = {
    'target_letters': [],  # List of 3 letters
    'current_index': 0,    # Which letter we're on (0, 1, or 2)
    'verified': False,
    'attempts': 0,
    'start_time': None,
    'frame_count': 0,
    'recent_predictions': [],
    'current_letter': None,
    'letters_completed': []  # Track completed letters
}

def generate_new_captcha():
    # Generate 3 unique random letters
    letters = random.sample(CAPTCHA_LETTERS, CAPTCHA_NUM_LETTERS)
    captcha_state['target_letters'] = letters
    captcha_state['current_index'] = 0
    captcha_state['verified'] = False
    captcha_state['start_time'] = time.time()
    captcha_state['frame_count'] = 0
    captcha_state['recent_predictions'] = []
    captcha_state['current_letter'] = None
    captcha_state['letters_completed'] = []
    return letters

def generate_captcha_frames():
    global captcha_state
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Get current target letter
        target = None
        if captcha_state['target_letters'] and captcha_state['current_index'] < len(captcha_state['target_letters']):
            target = captcha_state['target_letters'][captcha_state['current_index']]
        
        predicted_letter = ""
        confidence = 0.0
        
        if results.hand_landmarks and target and not captcha_state['verified']:
            hand = results.hand_landmarks[0]
            draw_hand_skeleton(frame, hand)
            
            landmark_row = []
            for lm in hand:
                landmark_row.extend([lm.x, lm.y, lm.z])
            
            normalized = normalize_landmarks(landmark_row)
            landmark_array = normalized.reshape(1, -1)
            
            prediction = model.predict(landmark_array)
            probabilities = model.predict_proba(landmark_array)
            
            raw_letter = prediction[0]
            raw_confidence = max(probabilities[0])
            
            captcha_state['recent_predictions'].append((raw_letter, raw_confidence))
            if len(captcha_state['recent_predictions']) > 5:
                captcha_state['recent_predictions'].pop(0)
            
            predicted_letter, confidence = get_smoothed_prediction(captcha_state['recent_predictions'])
            
            # Verification logic for current letter
            if predicted_letter == target and confidence >= CAPTCHA_CONFIDENCE:
                if captcha_state['current_letter'] == target:
                    captcha_state['frame_count'] += 1
                    if captcha_state['frame_count'] >= CAPTCHA_FRAMES_TO_VERIFY:
                        # Letter verified! Move to next
                        captcha_state['letters_completed'].append(target)
                        captcha_state['current_index'] += 1
                        captcha_state['frame_count'] = 0
                        captcha_state['recent_predictions'] = []
                        captcha_state['current_letter'] = None
                        
                        # Check if all 3 letters completed
                        if captcha_state['current_index'] >= CAPTCHA_NUM_LETTERS:
                            captcha_state['verified'] = True
                else:
                    captcha_state['current_letter'] = target
                    captcha_state['frame_count'] = 1
            else:
                captcha_state['frame_count'] = 0
                captcha_state['current_letter'] = None
        
        # UI Overlay - Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 40), -1)
        cv2.putText(frame, "SECURE THE QUEUE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"Sign 3 Letters - {captcha_state['current_index']}/3 Complete", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Show all 3 letters at top right
        if captcha_state['target_letters']:
            letters_display = ""
            for i, letter in enumerate(captcha_state['target_letters']):
                if i < captcha_state['current_index']:
                    letters_display += f"[{letter}✓] "  # Completed
                elif i == captcha_state['current_index']:
                    letters_display += f"[{letter}] "  # Current
                else:
                    letters_display += f" {letter}  "  # Upcoming
            cv2.putText(frame, letters_display, (frame.shape[1] - 250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        if captcha_state['verified']:
            # All 3 verified!
            cv2.rectangle(frame, (0, 80), (frame.shape[1], 180), (0, 100, 0), -1)
            cv2.putText(frame, "ALL 3 VERIFIED!", (frame.shape[1]//2 - 120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        elif target:
            # Show current target letter
            cv2.putText(frame, f"NOW: {target}", (frame.shape[1]//2 - 80, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 255), 3)
            
            # Show progress indicator
            progress_text = f"Letter {captcha_state['current_index'] + 1} of 3"
            cv2.putText(frame, progress_text, (frame.shape[1]//2 - 60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            if predicted_letter:
                color = (0, 255, 0) if predicted_letter == target else (0, 0, 255)
                cv2.putText(frame, f"Detecting: {predicted_letter}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Progress bar for current letter
            progress = min(captcha_state['frame_count'] / CAPTCHA_FRAMES_TO_VERIFY, 1.0)
            bar_y = frame.shape[0] - 40
            cv2.rectangle(frame, (50, bar_y), (frame.shape[1]-50, bar_y+20), (50, 50, 50), -1)
            bar_width = int((frame.shape[1]-100) * progress)
            bar_color = (0, 255, 0) if predicted_letter == target else (100, 100, 100)
            cv2.rectangle(frame, (50, bar_y), (50 + bar_width, bar_y+20), bar_color, -1)
            
            # Show completed letters
            if captcha_state['letters_completed']:
                completed_text = "Done: " + " ".join(captcha_state['letters_completed'])
                cv2.putText(frame, completed_text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==================== ROUTES ====================

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/translator')
def translator():
    return render_template('translator.html')

@app.route('/translator_feed')
def translator_feed():
    return Response(generate_translator_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_translator_text')
def get_translator_text():
    return jsonify({'text': translator_state['typed_text']})

@app.route('/clear_translator_text')
def clear_translator_text():
    translator_state['typed_text'] = ''
    return jsonify({'status': 'cleared'})

@app.route('/translator_backspace')
def translator_backspace():
    translator_state['typed_text'] = translator_state['typed_text'][:-1]
    return jsonify({'text': translator_state['typed_text']})

@app.route('/translator_add_space')
def translator_add_space():
    translator_state['typed_text'] += ' '
    return jsonify({'text': translator_state['typed_text']})

@app.route('/correct_translator_text')
def correct_translator_text():
    translator_state['typed_text'] = simple_correct(translator_state['typed_text'])
    return jsonify({'text': translator_state['typed_text']})

@app.route('/captcha')
def captcha():
    return render_template('captcha.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_captcha_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_captcha')
def start_captcha():
    letters = generate_new_captcha()
    return jsonify({
        'status': 'started', 
        'target_letters': letters,
        'current_index': 0,
        'timeout': CAPTCHA_TIMEOUT
    })

@app.route('/check_status')
def check_status():
    return jsonify({
        'verified': captcha_state['verified'],
        'target_letters': captcha_state['target_letters'],
        'current_index': captcha_state['current_index'],
        'letters_completed': captcha_state['letters_completed'],
        'attempts': captcha_state['attempts']
    })

@app.route('/new_challenge')
def new_challenge():
    captcha_state['attempts'] += 1
    if captcha_state['attempts'] >= 3:
        return jsonify({'status': 'locked', 'message': 'Too many attempts.'})
    letters = generate_new_captcha()
    return jsonify({
        'status': 'new_challenge', 
        'target_letters': letters,
        'current_index': 0
    })

@app.route('/success')
def success():
    if captcha_state['verified']:
        return render_template('success.html')
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)