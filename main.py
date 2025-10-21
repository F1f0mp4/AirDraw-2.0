import cv2
import mediapipe as mp
import math

# --- CONFIGURATION ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PINCH_THRESHOLD = 0.05  # Distance between thumb & index to consider as "pinch"
DRAW_COLOR = (0, 255, 0)  # Green drawing color
BRUSH_SIZE = 6

# --- MEDIAPIPE HANDS SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CAMERA INIT ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

cv2.namedWindow('Air Draw', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Air Draw', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Air Draw', 100, 100)

print("🎨 Air Drawing Activated")
print("📋 Controls:")
print("   - Pinch thumb + index to draw")
print("   - Press 'C' to clear canvas")
print("   - Press 'Q' to quit")

# --- INITIAL STATE ---
drawing = False
prev_x, prev_y = None, None
canvas = None

# --- HAND TRACKING LOOP ---
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Ignoring empty frame.")
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = frame.copy() * 0  # Black transparent canvas

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get thumb and index coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert normalized coords to pixel coords
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Compute distance
                distance = math.hypot(x2 - x1, y2 - y1)

                # Draw visual markers
                cv2.circle(frame, (x1, y1), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (255, 0, 255), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Calculate midpoint (where we draw)
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

                # --- PINCH TO DRAW ---
                if distance < PINCH_THRESHOLD * w:
                    if not drawing:
                        drawing = True
                        prev_x, prev_y = mid_x, mid_y
                        print("🖊 Drawing started")

                    # Draw line from previous to current point
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (mid_x, mid_y), DRAW_COLOR, BRUSH_SIZE)
                    prev_x, prev_y = mid_x, mid_y
                else:
                    if drawing:
                        drawing = False
                        prev_x, prev_y = None, None
                        print("✋ Drawing stopped")

        # Combine frame + canvas overlay
        combined = cv2.addWeighted(frame, 1, canvas, 1, 0)

        # --- DISPLAY ---
        cv2.putText(combined, "Press 'C' to clear | 'Q' to quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Air Draw', combined)

        # --- KEY CONTROLS ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = frame.copy() * 0
            print("🧹 Canvas cleared")

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("✅ Exited cleanly.")
