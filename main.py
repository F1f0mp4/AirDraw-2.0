import cv2
import mediapipe as mp
import numpy as np
import math
import time

# === CONFIGURATION ===
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PINCH_THRESHOLD = 0.05
TOOL_SIZE = 8
MIN_SIZE, MAX_SIZE = 2, 60

# === COLORS (BGR for OpenCV) ===
BASIC_COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255),
}
EXTENDED_COLORS = {
    "Purple": (128, 0, 128),
    "Cyan": (255, 255, 0),
    "Orange": (0, 165, 255),
    "Pink": (203, 192, 255),
    "Gray": (128, 128, 128),
}

# --- State Variables ---
expanded = False
selected_color = BASIC_COLORS["Green"]
last_toggle_time = 0 # Used to prevent rapid palette toggling

# === MEDIAPIPE SETUP ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === CAMERA INIT ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

cv2.namedWindow("AirDraw", cv2.WINDOW_NORMAL)
canvas = None
prev_x, prev_y = None, None

left_y_start = None  # for left-hand pinch resize

print("🎨 AirDraw 5.2 | Multi-Hand Edition")
print("Controls:")
print("  🖐 Right Hand → Point to Draw / Open Palm to Erase")
print("  🤏 Right Hand → Pinch on UI to select color or expand palette")
print("  ↕️ Left Hand → Pinch & Move Up/Down to adjust tool size")
print("  🧹 Press 'C' to clear, 'Q' to quit\n")


# === UTILITIES ===
def fingers_up(lm):
    """Checks which fingers are extended."""
    tips = [4, 8, 12, 16, 20]
    pip = [2, 6, 10, 14, 18]
    fingers = []
    # Thumb: Compare x-coordinates of tip and pip
    fingers.append(lm[tips[0]].x < lm[pip[0]].x if lm[tips[0]].x < lm[tips[0]-1].x else lm[tips[0]].x > lm[pip[0]].x)
    # Other 4 fingers: Compare y-coordinates
    for i in range(1, 5):
        fingers.append(lm[tips[i]].y < lm[pip[i]].y)
    return fingers

def draw_palette(frame, finger_x=None, finger_y=None):
    """
    Draws a sleek two-row color palette with a clear, interactive expand/collapse button.
    """
    global expanded, selected_color

    # --- Visual Constants ---
    base_y = 10
    color_box_size = 55
    spacing = 15
    label_offset = 25
    palette_start_y = 40 # Y position for the first row of colors

    # --- Colors to Display ---
    color_sets = [("Basic", BASIC_COLORS)]
    if expanded:
        color_sets.append(("Extended", EXTENDED_COLORS))

    y = palette_start_y
    for set_name, color_dict in color_sets:
        x = spacing
        for name, color in color_dict.items():
            pt1 = (x, y)
            pt2 = (x + color_box_size, y + color_box_size)
            cv2.rectangle(frame, pt1, pt2, color, -1)

            # --- Visual Feedback ---
            # Outline selected color
            if color == selected_color:
                cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 4)

            # Show hover effect
            if finger_x and finger_y and pt1[0] < finger_x < pt2[0] and pt1[1] < finger_y < pt2[1]:
                 cv2.rectangle(frame, pt1, pt2, (200, 200, 200), 3)

            x += color_box_size + spacing
        y += color_box_size + spacing + 10 # Move to the next row

    # --- Expand/Collapse Button ---
    expand_button_x, expand_button_y = WINDOW_WIDTH - 260, base_y
    expand_button_w, expand_button_h = 250, 50
    button_color = (80, 80, 80)
    
    # Hover effect for the button
    if finger_x and finger_y and \
       expand_button_x < finger_x < expand_button_x + expand_button_w and \
       expand_button_y < finger_y < expand_button_y + expand_button_h:
        button_color = (120, 120, 120)

    cv2.rectangle(frame, (expand_button_x, expand_button_y), (expand_button_x + expand_button_w, expand_button_y + expand_button_h), button_color, -1)
    cv2.rectangle(frame, (expand_button_x, expand_button_y), (expand_button_x + expand_button_w, expand_button_y + expand_button_h), (200, 200, 200), 2)

    expand_text = "▲ Collapse" if expanded else "▼ Expand"
    text_size = cv2.getTextSize(expand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = expand_button_x + (expand_button_w - text_size[0]) // 2
    text_y = expand_button_y + (expand_button_h + text_size[1]) // 2
    cv2.putText(frame, expand_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def handle_palette_interaction(x, y):
    """
    Handles clicks on the palette. Determines if a color or the expand button was selected.
    This function's layout logic MUST perfectly match `draw_palette`.
    """
    global expanded, selected_color, last_toggle_time

    # --- Check for Expand/Collapse click ---
    base_y = 10
    expand_button_x, expand_button_y = WINDOW_WIDTH - 260, base_y
    expand_button_w, expand_button_h = 250, 50

    # Debounce to prevent flickering
    if time.time() - last_toggle_time > 0.5:
        if expand_button_x < x < expand_button_x + expand_button_w and \
           expand_button_y < y < expand_button_y + expand_button_h:
            expanded = not expanded
            last_toggle_time = time.time()
            return # Exit after toggling

    # --- Check for color selection ---
    color_box_size = 55
    spacing = 15
    palette_start_y = 40

    color_sets = [("Basic", BASIC_COLORS)]
    if expanded:
        color_sets.append(("Extended", EXTENDED_COLORS))

    current_y = palette_start_y
    for set_name, color_dict in color_sets:
        current_x = spacing
        for name, color in color_dict.items():
            pt1 = (current_x, current_y)
            pt2 = (current_x + color_box_size, current_y + color_box_size)
            if pt1[0] < x < pt2[0] and pt1[1] < y < pt2[1]:
                selected_color = color
                return # Exit after finding a color
            current_x += color_box_size + spacing
        current_y += color_box_size + spacing + 10


def hud(frame, mode, size):
    """Draws the Heads-Up Display with current mode and tool size."""
    cv2.putText(frame, f"Mode: {mode}", (20, WINDOW_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Size: {int(size)}px", (250, WINDOW_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.rectangle(frame, (WINDOW_WIDTH - 80, WINDOW_HEIGHT - 60), (WINDOW_WIDTH - 20, WINDOW_HEIGHT-20), selected_color, -1)
    cv2.rectangle(frame, (WINDOW_WIDTH - 80, WINDOW_HEIGHT - 60), (WINDOW_WIDTH - 20, WINDOW_HEIGHT-20), (255,255,255), 2)
    cv2.putText(frame, "Color", (WINDOW_WIDTH - 78, WINDOW_HEIGHT-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# === MAIN LOOP ===
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    mode = "Idle"
    right_hand_coords = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lm = hand_landmarks.landmark
            label = handedness.classification[0].label

            index_tip = lm[8]
            thumb_tip = lm[4]
            x_i, y_i = int(index_tip.x * w), int(index_tip.y * h)
            x_t, y_t = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            # Calculate distance between index and thumb for pinch detection
            dist = math.hypot(x_t - x_i, y_t - y_i)
            pinch = dist < (PINCH_THRESHOLD * w)
            fingers = fingers_up(lm)

            # === LEFT HAND: Size control ===
            if label == "Left":
                if pinch:
                    mode = "Adjust Size"
                    if left_y_start is None:
                        left_y_start = y_i
                    
                    # Calculate change in y and adjust tool size
                    dy = left_y_start - y_i
                    TOOL_SIZE = np.clip(TOOL_SIZE + dy * 0.1, MIN_SIZE, MAX_SIZE)
                    left_y_start = y_i
                    cv2.circle(frame, (x_i, y_i), 15, (0, 255, 0), -1)
                else:
                    left_y_start = None
                continue # Skip to next hand

            # === RIGHT HAND: Draw / Erase / Palette ===
            if label == "Right":
                right_hand_coords = (x_i, y_i)
                # --- Gesture Detection ---
                if fingers[1] and fingers[2] and not any(fingers[2:]):
                     # Two fingers up is a special case of pinch, handle UI first
                     pass

                # 1. Pinch for UI Interaction
                if pinch:
                    mode = "UI Mode"
                    handle_palette_interaction(x_i, y_i)
                    prev_x, prev_y = None, None
                
                # 2. Open Palm for Eraser
                elif fingers == [True, True, True, True, True]:
                    mode = "Eraser"
                    # Draw a visual indicator for the eraser
                    cv2.circle(frame, (x_i, y_i), int(TOOL_SIZE * 1.5), (50, 50, 50), -1)
                    cv2.circle(frame, (x_i, y_i), int(TOOL_SIZE * 1.5), (0, 0, 255), 2)
                    cv2.line(canvas, (prev_x, prev_y), (x_i, y_i), (0, 0, 0), int(TOOL_SIZE * 3.0)) if prev_x else None
                    prev_x, prev_y = x_i, y_i

                # 3. Index Finger for Drawing
                elif fingers[1] and not any(fingers[2:]):
                    mode = "Drawing"
                    # Draw a cursor to show where you are drawing
                    cv2.circle(frame, (x_i, y_i), int(TOOL_SIZE / 2), selected_color, -1)
                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x_i, y_i), selected_color, int(TOOL_SIZE))
                    prev_x, prev_y = x_i, y_i

                # 4. Any other gesture stops drawing
                else:
                    prev_x, prev_y = None, None

    # === Combine layers for solid colors using a mask ===
    # 1. Create a mask of the drawing (anything not black on the canvas)
    img2gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Threshold the mask: pixels > 10 are part of the drawing
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 2. Black-out the area of the drawing in the camera feed
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # 3. Get only the drawing from the canvas
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    # 4. Combine the background frame and the canvas foreground for an opaque result
    combined = cv2.add(frame_bg, canvas_fg)
    
    # === Draw UI on top of the combined image ===
    # Pass right hand coords for hover effects, or None if not present
    finger_tip_for_hover = right_hand_coords if right_hand_coords else (None, None)
    draw_palette(combined, finger_tip_for_hover[0], finger_tip_for_hover[1])
    
    hud(combined, mode, TOOL_SIZE)
    cv2.imshow("AirDraw", combined)

    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas = np.zeros_like(frame)
        print("🧹 Canvas cleared")

cap.release()
cv2.destroyAllWindows()
print("✅ Session ended")

