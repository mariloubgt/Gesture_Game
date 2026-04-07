import cv2
import random
import time
import numpy as np

cap = cv2.VideoCapture(0)

roi_top, roi_bottom = 100, 400
roi_left, roi_right = 300, 600

cv2.namedWindow("Gesture Game", cv2.WINDOW_NORMAL)
cv2.moveWindow("Gesture Game", 100, 100)
cv2.resizeWindow("Gesture Game", 800, 600)

player_score = 0 
computer_score = 0
last_round_time = time.time()
round_delay = 3 
current_gesture = ""
computer_choice = ""
result = ""

def get_winner(player, computer):
    if player == computer:
        return "Tie"
    wins = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    return "You win!" if wins[player] == computer else "Computer wins!"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # Extract the ROI
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range in HSV
    lower_skin = (0, 20, 70)
    upper_skin = (20, 255, 255)

    # Create the mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # --- STAGE 3: Clean the mask ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # --- STAGE 3: Find the hand contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [hand_contour], -1, (0, 0, 255), 2)

        # --- STAGE 4: Convex hull and defect counting ---
        if cv2.contourArea(hand_contour) > 5000:

            hull = cv2.convexHull(hand_contour, returnPoints=False)
            hull_points = cv2.convexHull(hand_contour, returnPoints=True)
            cv2.drawContours(roi, [hull_points], -1, (0, 255, 0), 2)

            defects = cv2.convexityDefects(hand_contour, hull)

            defect_count = 0

            # --- FIX 2: angle filter ---
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0

                    # FIX 1: raised depth threshold
                    if depth > 35:
                        start = tuple(hand_contour[s][0])
                        end   = tuple(hand_contour[e][0])
                        far   = tuple(hand_contour[f][0])

                        # Calculate angle at the defect point
                        a = ((end[0]-start[0])**2 + (end[1]-start[1])**2) ** 0.5
                        b = ((far[0]-start[0])**2  + (far[1]-start[1])**2) ** 0.5
                        c = ((end[0]-far[0])**2    + (end[1]-far[1])**2)   ** 0.5

                        angle = (b**2 + c**2 - a**2) / (2*b*c + 1e-6)

                        # Only count if angle < 90 degrees (real finger gap)
                        if angle < 0.3:
                            defect_count += 1
                            cv2.circle(roi, far, 5, (0, 0, 255), -1)

            # --- FIX 3: solidity check for rock ---
            area = cv2.contourArea(hand_contour)
            hull_area = cv2.contourArea(hull_points)
            solidity = area / hull_area if hull_area > 0 else 0

            # --- Classify the gesture ---
            if defect_count == 0 or solidity > 0.9:
                gesture = "Rock"
            elif 1 <= defect_count <= 3:
                gesture = "Scissors"
            else:
                gesture = "Paper"

            cv2.putText(frame, f"Gesture: {gesture}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            cv2.putText(frame, f"Defects: {defect_count}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Solidity: {solidity:.2f}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # --- STAGE 5: Game logic with timer ---
            current_gesture = gesture
            now = time.time()

            if now - last_round_time >= round_delay:
                computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                result = get_winner(current_gesture, computer_choice)

                if result == "You win!":
                    player_score += 1
                elif result == "Computer wins!":
                    computer_score += 1

                last_round_time = now

            # Display game info
            cv2.putText(frame, f"Computer: {computer_choice}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Result: {result}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Score  You: {player_score}  PC: {computer_score}", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the mask in a separate window
    cv2.imshow("Skin Mask", mask)

    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here", (roi_left, roi_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gesture Game", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()