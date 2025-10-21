import cv2
import mediapipe as mp
import math
import numpy as np

# --- CONFIGURATION ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PINCH_THRESHOLD = 0.05  # distance (normalized) for pinch
BRUSH_SIZE = 8
ERASER_SIZE = 50

# --- COLORS (BGR format for OpenCV) ---
COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255)
}
color_names = list(COLORS.keys())
selected_color = COLORS["Green"]

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CAMERA INIT ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

cv2.namedWindow('AirDraw', cv2.WINDOW_NORMAL)
cv2.resizeWindow('AirDraw', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('AirDraw', 100, 100)

print("🎨 AirDraw 2.0 launched")
print("📋 Gestures:")
print("   ✏️ Index finger = draw")
print("   🤏 Pinch = select color (top bar)")
print("   ✋ Open hand = erase")

canvas = None
prev_x, prev_y = None, None
mode = "draw"

# --- HELPER FUNCTIONS ---
def count_fingers(lm):
    fingers = []
    # Thumb
    fingers.append(lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x)
    # Other 4 fingers
    fingers.append(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    fingers.append(lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    fingers.append(lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y)
    fingers.append(lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y)
    return sum(fingers)

def draw_color_bar(frame):
    """Draw color palette bar at top of screen"""
    bar_height = 60
    step = WINDOW_WIDTH // len(COLORS)
    for i, name in enumerate(color_names):
        color = COLORS[name]
        x1, x2 = i * step, (i + 1) * step
        cv2.rectangle(frame, (x1, 0), (x2, bar_height), color, -1)
        if color == selected_color:
            cv2.rectangle(frame, (x1, 0), (x2, bar_height), (255, 255, 255), 3)
        cv2.putText(frame, name, (x1 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# --- HAND TRACKING LOOP ---
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros_like(frame)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                # Landmark points
                index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

                x_i, y_i = int(index_tip.x * w), int(index_tip.y * h)
                x_t, y_t = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Finger logic
                finger_count = count_fingers(lm)
                distance = math.hypot(x_t - x_i, y_t - y_i)
                pinch = distance < PINCH_THRESHOLD * w
                index_up = index_tip.y < index_pip.y

                # --- Gesture Logic ---
                if finger_count == 5:
                    mode = "erase"
                    cv2.putText(frame, "✋ Eraser Mode", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.circle(frame, (x_i, y_i), ERASER_SIZE // 2, (0, 0, 255), 2)
                    cv2.circle(canvas, (x_i, y_i), ERASER_SIZE, (0, 0, 0), -1)
                    prev_x, prev_y = None, None

                elif pinch:
                    mode = "select"
                    draw_color_bar(frame)
                    bar_height = 60
                    if y_i < bar_height:  # if pinch is inside color bar
                        step = WINDOW_WIDTH // len(COLORS)
                        index = min(x_i // step, len(COLORS) - 1)
                        selected_color = COLORS[color_names[index]]
                    cv2.putText(frame, "🎯 Selecting Color", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, selected_color, 3)
                    prev_x, prev_y = None, None

                elif index_up and not pinch and finger_count == 1:
                    mode = "draw"
                    cv2.putText(frame, "✏️ Drawing Mode", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, selected_color, 3)
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x_i, y_i), selected_color, BRUSH_SIZE)
                    prev_x, prev_y = x_i, y_i

                else:
                    prev_x, prev_y = None, None

        # --- Combine Frame + Canvas ---
        combined = cv2.addWeighted(frame, 1, canvas, 1, 0)
        draw_color_bar(combined)
        cv2.imshow('AirDraw', combined)

        # --- Keyboard Controls ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            print("🧹 Canvas cleared")

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("✅ Session Ended.")
