import cv2
import mediapipe as mp
import math

# --- CONFIGURATION ---
WINDOW_WIDTH, WINDOW_HEIGHT = 720, 450
PINCH_THRESHOLD = 0.05  # smaller = more sensitive (distance between thumb & index)

# --- MEDIAPIPE HANDS SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- CAMERA INIT ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

cv2.namedWindow('Finger Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Finger Tracking', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Finger Tracking', 100, 100)

print("🖐 Starting finger tracking...")
print("📋 Controls:")
print("   - Press 'Q' to quit")

# --- HAND TRACKING LOOP ---
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Ignoring empty frame.")
            continue

        # Mirror and convert to RGB
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # --- VISUALIZATION ---
        annotated = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # --- PINCH DETECTION ---
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Compute normalized distance between thumb & index
                dx = thumb_tip.x - index_tip.x
                dy = thumb_tip.y - index_tip.y
                distance = math.sqrt(dx * dx + dy * dy)

                # --- Check for pinch gesture ---
                if distance < PINCH_THRESHOLD:
                    cv2.putText(annotated, "🤏 PINCH DETECTED!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                else:
                    cv2.putText(annotated, "No Pinch", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # --- DISPLAY FRAME ---
        cv2.imshow('Finger Tracking', annotated)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
print("👋 Exiting...")
cap.release()
cv2.destroyAllWindows()
print("✅ Done.")
