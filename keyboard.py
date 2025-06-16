import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize TTS engine
engine = pyttsx3.init()

# Keyboard configuration
keyboard_keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['READ', 'DELETE', 'BACKSPACE', 'SPACE']
]

# Key dimensions and settings
key_size = 70  # Reduced key size for alphabet buttons
padding = 15  # Padding between keys
special_key_size = 100  # Increased size for special keys
special_key_padding = 40  # Increased padding for special keys (more space)
left_padding = 150  # Padding for left space
keyboard_origin = (50 + left_padding, 150)  # Adjusted keyboard origin with left padding and higher position
typed_text = ""
pinched = False
last_key_time = 0  # Tracks time of last key press
cooldown_time = 0.3  # Minimum time between key presses
pinch_threshold = 0.05  # Threshold for pinch detection
pressed_button = None  # Tracks which button is currently pressed


# Function to draw the keyboard
def draw_keyboard(frame):
    global pressed_button

    for row_idx, row in enumerate(keyboard_keys):
        for col_idx, key in enumerate(row):
            key_size_to_use = special_key_size if key in ['READ', 'DELETE', 'BACKSPACE', 'SPACE'] else key_size
            key_padding_to_use = special_key_padding if key in ['READ', 'DELETE', 'BACKSPACE', 'SPACE'] else padding

            top_left = (keyboard_origin[0] + col_idx * (key_size_to_use + key_padding_to_use),
                        keyboard_origin[1] + row_idx * (key_size_to_use + key_padding_to_use))
            bottom_right = (top_left[0] + key_size_to_use, top_left[1] + key_size_to_use)

            # Draw rectangle with rounded corners
            radius = 15  # Border radius for keys
            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)
            cv2.ellipse(frame, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90,
                        (255, 255, 255), -1)
            cv2.ellipse(frame, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 270, 0, 90,
                        (255, 255, 255), -1)
            cv2.ellipse(frame, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 90, 0, 90,
                        (255, 255, 255), -1)
            cv2.ellipse(frame, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90,
                        (255, 255, 255), -1)
            cv2.rectangle(frame, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]),
                          (255, 255, 255), -1)
            cv2.rectangle(frame, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius),
                          (255, 255, 255), -1)

            # Highlight key when pressed
            if pressed_button == key:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), -1)  # Green for hover effect

            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = top_left[0] + (key_size_to_use - text_size[0]) // 2
            text_y = top_left[1] + (key_size_to_use + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


# Background TTS function
def speak_text_async(text):
    threading.Thread(target=speak_text, args=(text,)).start()


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


# Detect the key being pressed
def detect_key(frame, landmarks):
    global typed_text, pinched, last_key_time, pressed_button

    # Get thumb and index finger tips
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    # Calculate pinch distance (between thumb and index finger tips)
    pinch_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) -
                                    np.array([index_tip.x, index_tip.y]))

    # Check if the pinch is detected and distance is small enough to indicate a pinch
    if pinch_distance < pinch_threshold and not pinched:
        pinched = True
        # Prevent multiple key presses in quick succession (debounce)
        current_time = time.time()
        if current_time - last_key_time > cooldown_time:
            last_key_time = current_time  # Update the last key press time

            # Convert normalized landmarks to screen coordinates
            x = int(index_tip.x * frame.shape[1])
            y = int(index_tip.y * frame.shape[0])

            # Add a slight buffer around each key for better accuracy
            buffer_size = 15  # Increased buffer for touch area
            for row_idx, row in enumerate(keyboard_keys):
                for col_idx, key in enumerate(row):
                    key_size_to_use = special_key_size if key in ['READ', 'DELETE', 'BACKSPACE', 'SPACE'] else key_size
                    key_padding_to_use = special_key_padding if key in ['READ', 'DELETE', 'BACKSPACE',
                                                                        'SPACE'] else padding

                    key_top_left = (keyboard_origin[0] + col_idx * (key_size_to_use + key_padding_to_use) - buffer_size,
                                    keyboard_origin[1] + row_idx * (key_size_to_use + key_padding_to_use) - buffer_size)
                    key_bottom_right = (key_top_left[0] + key_size_to_use + 2 * buffer_size,
                                        key_top_left[1] + key_size_to_use + 2 * buffer_size)

                    # Check if the finger is within the bounds of the key
                    if key_top_left[0] <= x <= key_bottom_right[0] and key_top_left[1] <= y <= key_bottom_right[1]:
                        pressed_button = key
                        if key == "READ":
                            speak_text_async(typed_text)
                        elif key == "DELETE":
                            typed_text = ""
                        elif key == "BACKSPACE":
                            typed_text = typed_text[:-1]
                        elif key == "SPACE":
                            typed_text += " "
                        else:
                            typed_text += key
                            speak_text_async(key)
                        return
    elif pinch_distance >= pinch_threshold:
        pinched = False
        pressed_button = None  # Reset pressed button


# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1100, 768))  # Force the camera to be 1024x768

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    draw_keyboard(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_key(frame, hand_landmarks.landmark)

    cv2.putText(frame, f"Typed: {typed_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
