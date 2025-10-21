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
    bar_y = 60
    radius = 25
    spacing = WINDOW_WIDTH // (len(COLORS) + 1)
    for i, name in enumerate(color_names):
        color = COLORS[name]
        cx = spacing * (i + 1)
        cy = bar_y
        # Draw color circle
        cv2.circle(frame, (cx, cy), radius, color, -1)
        # Highlight selected color
        if color == selected_color:
            cv2.circle(frame, (cx, cy), radius + 5, (255, 255, 255), 3)
        cv2.putText(frame, name, (cx - 35, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


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
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

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
                    if y_i < bar_height:
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
                        cv2.line(canvas, (prev_x, prev_y), (x_i, y_i), selected_color, BRUSH_SIZE, cv2.LINE_8)
                    prev_x, prev_y = x_i, y_i

                else:
                    prev_x, prev_y = None, None

        # --- Combine Frame + Canvas (Solid Color Merge) ---
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        draw_color_bar(combined)
        cv2.imshow('AirDraw', combined)

        # --- Keyboard Controls ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame, dtype=np.uint8)
            print("🧹 Canvas cleared")

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("✅ Session Ended.")
