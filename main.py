import cv2
import numpy as np
import os
from models.classical import ClassicalBackend

# Upper render bound: a bit below the ROI horizon (y=250) so we don't draw
# the carpet in a region where the quadratic fit is barely supported by data.
RENDER_Y_TOP = 500
RENDER_CONF_THRESHOLD = 0.35

def render_normal_overlay(frame, data):
    h, w = frame.shape[:2]

    # Draw the green carpet only when the fit is trustworthy.
    if data.confidence >= RENDER_CONF_THRESHOLD:
        overlay = frame.copy()
        plot_y = np.linspace(RENDER_Y_TOP, h - 1, 30)
        l_x = data.left_fit[0]*plot_y**2 + data.left_fit[1]*plot_y + data.left_fit[2]
        r_x = data.right_fit[0]*plot_y**2 + data.right_fit[1]*plot_y + data.right_fit[2]

        # Clamp to image bounds so a noisy polynomial can't draw off-screen.
        l_x = np.clip(l_x, 0, w - 1)
        r_x = np.clip(r_x, 0, w - 1)

        pts_left = np.array([np.transpose(np.vstack([l_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)
    
    # --- ADD THE HUD HERE ---
    frame = draw_description(frame, data)
    
    return frame

def draw_description(img, data):
    """
    Draws a semi-transparent HUD box in the top-right corner.
    """
    h, w = img.shape[:2]
    
    # 1. Define Box Dimensions
    box_width = 310
    box_height = 100
    margin = 10
    
    # Calculate coordinates for top-right
    # Top-Left point of the box
    tr_x1 = w - box_width - margin
    tr_y1 = margin
    # Bottom-Right point of the box
    tr_x2 = w - margin
    tr_y2 = margin + box_height

    # 2. Draw Background
    overlay = img.copy()
    cv2.rectangle(overlay, (tr_x1, tr_y1), (tr_x2, tr_y2), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # 3. Add Text (Anchored to the new box position)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_x = tr_x1 + 10  # 10px padding inside the box
    
    status_text = "LANE TRACKED" if data.confidence >= RENDER_CONF_THRESHOLD else "SEARCHING..."
    status_color = (0, 255, 0) if data.confidence >= RENDER_CONF_THRESHOLD else (0, 0, 255)

    cv2.putText(img, status_text, (text_x, tr_y1 + 30), font, 0.7, status_color, 2)
    cv2.putText(img, f"Algorithm: {data.source.title()}", (text_x, tr_y1 + 60), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Conf: {data.confidence*100:.1f}%", (text_x, tr_y1 + 85), font, 0.6, (255, 255, 255), 1)
    
    return img

def create_debug_tile(frame, gray, edges, data):
    # 1. Convert Grayscale and Edges to 3-channel
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 2. Get the final result window (already has its HUD)
    res_window = render_normal_overlay(frame, data)
    
    # 3. Add Labels to each quadrant
    # Arguments: img, text, pos, font, scale, color, thickness
    cv2.putText(frame, "1. ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(gray_3ch, "2. GRAYSCALE + BLUR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(edges_3ch, "3. CANNY EDGES + ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(res_window, "4. FINAL PREDICTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 4. Stack: [Original, Gray] over [Edges, Result]
    top = np.hstack((frame, gray_3ch))
    bottom = np.hstack((edges_3ch, res_window))
    
    # Final combined image
    full_tile = np.vstack((top, bottom))
    
    # Resize so it fits nicely on a standard 1080p monitor
    return cv2.resize(full_tile, (1280, 720))

def main():
    cap = cv2.VideoCapture("test-videos/test_video_1.mp4")
    detector = ClassicalBackend()
    
    # State variable to track if we are paused
    is_paused = False

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            # Detection & Visualization
            lane_data, gray, edges = detector.process(frame)
            debug_view = create_debug_tile(frame, gray, edges, lane_data)
            cv2.imshow("Lane Detection Pipeline", debug_view)

        # Handle Key Presses
        key = cv2.waitKey(25) & 0xFF
        
        # Spacebar (key code 32) toggles pause
        if key == ord(' '): 
            is_paused = not is_paused
            print("Paused" if is_paused else "Resuming...")
            
        # 'q' quits the program
        elif key == ord('q'):
            break
            
        # If paused, we need a tiny delay so the CPU doesn't redline
        if is_paused:
            # We stay on the same frame until space is pressed again
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()