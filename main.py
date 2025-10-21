import cv2
import mediapipe as mp
import math
import numpy as np

# --- CONFIGURATION ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PINCH_THRESHOLD = 0.05
BRUSH_SIZE = 8
ERASER_SIZE = 50

# --- COLORS (BGR format) ---
COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255),
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Orange": (0, 165, 255),
    "Purple": (128, 0, 128),
    "Pink": (203, 192, 255)
}
color_names = list(COLORS.keys())
selected_color = COLORS["Green"]

# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands

# --- Camera Init ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

cv2.namedWindow('AirDraw', cv2.WINDOW_NORMAL)
cv2.resizeWindow('AirDraw', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('AirDraw', 100, 100)

print("🎨 AirDraw 3.0 launched")
print("📋 Gestures:")
print("   ✏️ Index finger = draw")
print("   🤏 Pinch = select color")
print("   ✋ Open hand = erase")

canvas = None
prev_x, prev_y = None, None
mode = "draw"

# --- Helper Functions ---
def count_fingers(lm):
    fingers = []
    fingers.append(lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x)
    fingers.append(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    fingers.append(lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    fingers.append(lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y)
    fingers.append(lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y)
    return sum(fingers)

def draw_palette(frame):
    """Draw circular color palette and UI buttons"""
    radius = 25
    spacing = WINDOW_WIDTH // (len(COLORS) + 2)
    y_pos = 60
    for i, name in enumerate(color_names):
        cx = spacing * (i + 1)
        cy = y_pos
        color = COLORS[name]
        cv2.circle(frame, (cx, cy), radius, color, -1)
        if color == selected_color:
            cv2.circle(frame, (cx, cy), radius + 5, (255, 255, 255), 3)
        cv2.putText(frame, name, (cx - 35, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw "Clear" button
    cv2.rectangle(frame, (20, 20), (120, 60), (0, 0, 255), -1)
    cv2.putText(frame, "Clear", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw "Eraser" button
    cv2.rectangle(frame, (130, 20), (230, 60), (128, 128, 128), -1)
    cv2.putText(frame, "Eraser", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_hud(frame):
    """Display active mode and color"""
    cv2.circle(frame, (80, 150), 30, selected_color, -1)
    cv2.putText(frame, f"Mode: {mode}", (130, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# --- Main Loop ---
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

                x_i, y_i = int(index_tip.x * w), int(index_tip.y * h)
                x_t, y_t = int(thumb_tip.x * w), int(thumb_tip.y * h)

                finger_count = count_fingers(lm)
                distance = math.hypot(x_t - x_i, y_t - y_i)
                pinch = distance < PINCH_THRESHOLD * w
                index_up = index_tip.y < index_pip.y

                # --- UI Interaction Zones ---
                if y_i < 60:  # top UI bar
                    spacing = WINDOW_WIDTH // (len(COLORS) + 2)
                    for i, name in enumerate(color_names):
                        cx = spacing * (i + 1)
                        cy = 60
                        if math.hypot(cx - x_i, cy - y_i) < 30:
                            selected_color = COLORS[name]
                            mode = "draw"
                            break

                # Clear button
                if 20 < x_i < 120 and 20 < y_i < 60 and pinch:
                    canvas = np.zeros_like(canvas)
                    print("🧹 Canvas cleared")

                # Eraser button
                if 130 < x_i < 230 and 20 < y_i < 60 and pinch:
                    mode = "erase"

                # --- Gesture Logic ---
                if finger_count == 5:
                    mode = "erase"
                    cv2.circle(frame, (x_i, y_i), ERASER_SIZE // 2, (0, 0, 255), 2)
                    cv2.circle(canvas, (x_i, y_i), ERASER_SIZE, (0, 0, 0), -1)
                    prev_x, prev_y = None, None

                elif index_up and not pinch and finger_count == 1:
                    mode = "draw"
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x_i, y_i), selected_color, BRUSH_SIZE, cv2.LINE_8)
                    prev_x, prev_y = x_i, y_i

                else:
                    prev_x, prev_y = None, None

        # --- Combine Canvas & Frame (solid merge) ---
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        draw_palette(combined)
        draw_hud(combined)
        cv2.imshow('AirDraw', combined)

        # --- Keyboard Controls ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("✅ Session Ended.")
