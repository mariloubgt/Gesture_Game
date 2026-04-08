import cv2
import pygame
import random
import time
import numpy as np
import sys
import math
from collections import deque, Counter

# ── INIT ────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

WIDTH, HEIGHT = 1100, 620
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors — Gesture Game")

# ── COLORS ──────────────────────────────────────────────────
BG        = (12, 14, 28)
PANEL     = (28, 32, 58)
GREEN     = (72, 214, 128)
YELLOW    = (250, 214, 68)
BLUE      = (88, 168, 255)
WHITE     = (245, 246, 252)
GRAY      = (150, 154, 178)
DARK_GRAY = (38, 42, 72)
RED       = (255, 92, 108)
ORANGE    = (255, 152, 72)
ACCENT    = (255, 96, 148)
PURPLE    = (168, 112, 255)
GLOW      = (60, 200, 255)
CV_CYAN   = (255, 220, 80)    # BGR for OpenCV drawing

# ── FONTS ───────────────────────────────────────────────────
font_huge  = pygame.font.SysFont("Segoe UI", 88, bold=True)
font_big   = pygame.font.SysFont("Segoe UI", 50, bold=True)
font_med   = pygame.font.SysFont("Segoe UI", 30, bold=True)
font_small = pygame.font.SysFont("Segoe UI", 21)
font_tiny  = pygame.font.SysFont("Segoe UI", 16)

# ── SOUNDS ──────────────────────────────────────────────────
def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None

snd_win  = load_sound("assets/win.wav")
snd_lose = load_sound("assets/lose.wav")
snd_tie  = load_sound("assets/tie.wav")
snd_beep = load_sound("assets/beep.wav")

def play_snd(snd):
    if snd:
        snd.play()

# ── WEBCAM ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── ROI ─────────────────────────────────────────────────────
# Centered box the user places their hand in.
# Sized as fractions of the frame so it adapts to any resolution.
ROI_X_FRAC = 0.27   # left edge
ROI_Y_FRAC = 0.08   # top edge
ROI_W_FRAC = 0.46   # width
ROI_H_FRAC = 0.80   # height

def roi_rect(fh, fw):
    x = int(fw * ROI_X_FRAC)
    y = int(fh * ROI_Y_FRAC)
    w = int(fw * ROI_W_FRAC)
    h = int(fh * ROI_H_FRAC)
    return x, y, w, h

# ── GESTURE HISTORY ────────────────────────────────────────
gesture_history = deque(maxlen=15)

# ── REUSABLE SURFACES ──────────────────────────────────────
_flash_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# ── PARTICLES ───────────────────────────────────────────────
particles = []

def spawn_particles(cx, cy, color, count=40):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        particles.append({
            "x": cx, "y": cy,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": 1.0,
            "decay": random.uniform(0.02, 0.05),
            "size": random.randint(4, 10),
            "color": color,
        })

def update_draw_particles():
    global particles
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.2
        p["life"] -= p["decay"]
    particles = [p for p in particles if p["life"] > 0]
    for p in particles:
        alpha = int(p["life"] * 255)
        size = max(1, int(p["size"] * p["life"]))
        surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*p["color"], alpha), (size, size), size)
        screen.blit(surf, (int(p["x"]) - size, int(p["y"]) - size))

# ── SCREEN FLASH ────────────────────────────────────────────
flash_timer = 0
flash_color = (0, 0, 0)

def trigger_flash(color):
    global flash_timer, flash_color
    flash_timer = 12
    flash_color = color

def draw_flash():
    global flash_timer
    if flash_timer > 0:
        alpha = int((flash_timer / 12) * 80)
        _flash_surf.fill((*flash_color, alpha))
        screen.blit(_flash_surf, (0, 0))
        flash_timer -= 1

# ── SHAKE ───────────────────────────────────────────────────
shake_timer = 0
shake_offset = (0, 0)

def trigger_shake():
    global shake_timer
    shake_timer = 10

def update_shake():
    global shake_timer, shake_offset
    if shake_timer > 0:
        shake_offset = (random.randint(-6, 6), random.randint(-6, 6))
        shake_timer -= 1
    else:
        shake_offset = (0, 0)

# ── HELPERS ─────────────────────────────────────────────────
GESTURE_COLOR = {
    "Rock": ORANGE, "Paper": BLUE, "Scissors": ACCENT, "---": GRAY,
}
RESULT_COLOR = {
    "You win!": GREEN, "Computer wins!": RED, "Tie": YELLOW, "Show your hand!": GRAY,
}

def get_winner(player, computer):
    if player == computer:
        return "Tie"
    wins = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    return "You win!" if wins[player] == computer else "Computer wins!"

def draw_rounded_rect(surface, color, rect, radius=18, alpha=255):
    x, y, w, h = rect
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0, 0, w, h), border_radius=radius)
    surface.blit(s, (x, y))

def draw_text_centered(surface, text, font, color, cx, cy):
    r = font.render(text, True, color)
    surface.blit(r, r.get_rect(center=(cx, cy)))

def draw_text(surface, text, font, color, x, y):
    surface.blit(font.render(text, True, color), (x, y))

# ════════════════════════════════════════════════════════════
#  TRADITIONAL CV PIPELINE
# ════════════════════════════════════════════════════════════

def build_skin_mask(roi_bgr):
    """
    Skin detection using HSV color thresholding.
    Defines a skin-tone range in HSV space and creates a binary mask.
    """
    blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)

    # Convert to HSV and threshold for skin-tone range
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Primary skin hue range (covers most skin tones)
    mask1 = cv2.inRange(hsv, (0, 30, 50), (25, 255, 255))
    # Secondary range for reddish skin tones that wrap around H=180
    mask2 = cv2.inRange(hsv, (160, 30, 50), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def count_defects(contour, hull_indices):
    """
    Count convexity defects that represent valleys between extended fingers.
    Uses adaptive depth threshold + angle filter.
    Returns (count, list_of_far_points_for_drawing).
    """
    if hull_indices is None or len(hull_indices) < 3:
        return 0, []

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0, []

    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri < 1:
        return 0, []

    # Depth threshold: proportional to hand size so it scales with distance
    min_depth = max(12.0, peri * 0.02)

    count = 0
    far_pts = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth < min_depth:
            continue

        start = contour[s][0]
        end   = contour[e][0]
        far   = contour[f][0]

        # Triangle sides
        a = np.hypot(end[0] - start[0], end[1] - start[1])
        b = np.hypot(far[0] - start[0], far[1] - start[1])
        c = np.hypot(end[0] - far[0],   end[1] - far[1])

        if b < 1 or c < 1:
            continue

        # Angle at the far point (valley between two fingers)
        cos_angle = (b * b + c * c - a * a) / (2 * b * c)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        # Real finger valleys are roughly 20°–100°
        if angle_deg < 100:
            count += 1
            far_pts.append(tuple(far))

    return count, far_pts


def classify_gesture(defect_count, solidity, circularity):
    """
    Heuristic classification from convexity defect count:
      0 defects        → Rock  (closed fist)
      1–3 defects      → Scissors  (two fingers up)
      4+ defects       → Paper  (open palm)
    """
    if defect_count == 0:
        return "Rock"
    if defect_count <= 3:
        return "Scissors"
    return "Paper"


def stable_gesture(history):
    """Return the most common real gesture in recent history."""
    real = [g for g in history if g != "---"]
    if len(real) < 3:
        return "---"
    recent_real = [g for g in list(history)[-9:] if g != "---"]
    if not recent_real:
        return "---"
    best, cnt = Counter(recent_real).most_common(1)[0]
    if cnt >= 3:
        return best
    return "---"


# ── PROCESS FRAME ──────────────────────────────────────────
def process_frame(frame):
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rx, ry, rw, rh = roi_rect(fh, fw)
    roi = frame[ry:ry + rh, rx:rx + rw]

    if roi.size == 0:
        gesture_history.append("---")
        return frame, stable_gesture(gesture_history)

    mask = build_skin_mask(roi)
    roi_area = rw * rh

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = "---"
    debug_text = ""

    if contours:
        hand = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand)
        min_area = max(3000, roi_area * 0.05)

        if area > min_area:
            # Draw contour
            cv2.drawContours(roi, [hand], -1, (80, 200, 255), 2, cv2.LINE_AA)

            # Convex hull
            hull_pts = cv2.convexHull(hand, returnPoints=True)
            hull_idx = cv2.convexHull(hand, returnPoints=False)
            cv2.drawContours(roi, [hull_pts], -1, (80, 255, 140), 2, cv2.LINE_AA)

            hull_area = cv2.contourArea(hull_pts)
            solidity = area / hull_area if hull_area > 0 else 0

            peri = cv2.arcLength(hand, True)
            circularity = (4 * math.pi * area) / (peri * peri) if peri > 0 else 0

            defect_count, far_pts = count_defects(hand, hull_idx)

            # Draw defect points (valleys between fingers)
            for pt in far_pts:
                cv2.circle(roi, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)

            detected = classify_gesture(defect_count, solidity, circularity)
            debug_text = f"d={defect_count} sol={solidity:.2f}"

    gesture_history.append(detected)
    stable = stable_gesture(gesture_history)

    # Draw ROI rectangle
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (120, 220, 255), 2)

    # HUD text
    label = stable
    if debug_text:
        label += f"  [{debug_text}]"
    cv2.putText(frame, label, (rx, max(20, ry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 255), 2, cv2.LINE_AA)

    return frame, stable


# ── FRAME → PYGAME SURFACE ─────────────────────────────────
def frame_to_surface(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    return surf.copy()


# ════════════════════════════════════════════════════════════
#  START SCREEN
# ════════════════════════════════════════════════════════════
def start_screen():
    btn_w, btn_h = 248, 64
    btn_x = WIDTH // 2 - btn_w // 2
    btn_y = 398
    pulse = 0.0
    clock = pygame.time.Clock()

    while True:
        screen.fill(BG)
        pulse += 0.045

        hdr = pygame.Surface((WIDTH, 180), pygame.SRCALPHA)
        for row in range(180):
            a = int(40 * (1 - row / 180))
            pygame.draw.line(hdr, (*PURPLE, a), (0, row), (WIDTH, row))
        screen.blit(hdr, (0, 0))

        draw_text_centered(screen, "Rock  Paper  Scissors", font_big, YELLOW, WIDTH // 2, 148)
        draw_text_centered(screen, "Traditional CV Pipeline", font_med, WHITE, WIDTH // 2, 218)
        draw_text_centered(screen, "Place your hand inside the blue box — good lighting helps.",
                           font_small, GRAY, WIDTH // 2, 268)
        draw_text_centered(screen, "3s prepare after GO! — then 5s each round to lock your gesture.",
                           font_tiny, GRAY, WIDTH // 2, 298)

        labels = ["Rock", "Paper", "Scissors"]
        colors = [ORANGE, BLUE, ACCENT]
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            cx = WIDTH // 2 - 130 + i * 130
            cy = int(348 + 5 * math.sin(pulse + i * 1.1))
            draw_rounded_rect(screen, col, (cx - 56, cy - 28, 112, 56), radius=14, alpha=50)
            draw_text_centered(screen, lbl, font_small, col, cx, cy)

        mx, my = pygame.mouse.get_pos()
        hovered = btn_x < mx < btn_x + btn_w and btn_y < my < btn_y + btn_h
        btn_col = GREEN if hovered else DARK_GRAY
        draw_rounded_rect(screen, btn_col, (btn_x, btn_y, btn_w, btn_h), radius=16)
        draw_text_centered(screen, "Play", font_med, WHITE, WIDTH // 2, btn_y + btn_h // 2)
        draw_text_centered(screen, "Q — quit", font_tiny, GRAY, WIDTH // 2, 540)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cleanup(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cleanup(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and hovered:
                return

        pygame.display.flip()
        clock.tick(60)


# ════════════════════════════════════════════════════════════
#  COUNTDOWN (before prep phase)
# ════════════════════════════════════════════════════════════
def countdown_screen():
    clock = pygame.time.Clock()
    for label in ["3", "2", "1", "GO!"]:
        color = YELLOW if label != "GO!" else GREEN
        start = time.time()
        play_snd(snd_beep)

        while time.time() - start < 0.85:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                surf = frame_to_surface(frame)
                surf = pygame.transform.smoothscale(surf, (640, 480))
                screen.blit(surf, (10, 20))
            else:
                screen.fill(BG)

            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 130))
            screen.blit(overlay, (0, 0))

            elapsed = time.time() - start
            radius = int(88 + 18 * math.sin(elapsed * 8))
            pygame.draw.circle(screen, (*color, 55), (WIDTH // 2, HEIGHT // 2), radius)
            pygame.draw.circle(screen, color, (WIDTH // 2, HEIGHT // 2), radius, 3)
            draw_text_centered(screen, label, font_huge, color, WIDTH // 2, HEIGHT // 2 - 18)
            if label != "GO!":
                draw_text_centered(screen, "Game starting…", font_small, GRAY,
                                   WIDTH // 2, HEIGHT // 2 + 58)
            else:
                draw_text_centered(screen, "Then: prepare your gesture", font_small, GRAY,
                                   WIDTH // 2, HEIGHT // 2 + 58)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cleanup(); sys.exit()

            pygame.display.flip()
            clock.tick(60)


# ════════════════════════════════════════════════════════════
#  WINNER SCREEN
# ════════════════════════════════════════════════════════════
def winner_screen(player_score, computer_score):
    if player_score > computer_score:
        msg, col = "You win!", GREEN
        spawn_particles(WIDTH // 2, HEIGHT // 2, GREEN, 80)
        spawn_particles(WIDTH // 2, HEIGHT // 2, YELLOW, 60)
        play_snd(snd_win)
    elif computer_score > player_score:
        msg, col = "PC wins", RED
        trigger_shake()
        play_snd(snd_lose)
    else:
        msg, col = "It's a tie", YELLOW
        play_snd(snd_tie)

    btn_w, btn_h = 268, 58
    btn_x = WIDTH // 2 - btn_w // 2
    btn_y = 428
    clock = pygame.time.Clock()

    while True:
        screen.fill(BG)
        update_shake()
        ox, oy = shake_offset

        draw_text_centered(screen, "Game Over", font_med, GRAY, WIDTH // 2 + ox, 132 + oy)
        draw_text_centered(screen, msg, font_big, col, WIDTH // 2 + ox, 208 + oy)

        draw_rounded_rect(screen, PANEL, (WIDTH // 2 - 224, 268, 448, 112), radius=18)
        draw_text_centered(screen, f"You  {player_score}  —  {computer_score}  PC",
                           font_med, WHITE, WIDTH // 2, 324)

        update_draw_particles()
        draw_flash()

        mx, my = pygame.mouse.get_pos()
        hovered = btn_x < mx < btn_x + btn_w and btn_y < my < btn_y + btn_h
        draw_rounded_rect(screen, GREEN if hovered else DARK_GRAY,
                          (btn_x, btn_y, btn_w, btn_h), radius=14)
        draw_text_centered(screen, "Play again", font_med, WHITE, WIDTH // 2, btn_y + btn_h // 2)
        draw_text_centered(screen, "Q — quit", font_tiny, GRAY, WIDTH // 2, 540)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cleanup(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cleanup(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and hovered:
                return True

        pygame.display.flip()
        clock.tick(60)


# ════════════════════════════════════════════════════════════
#  GAME SCREEN
# ════════════════════════════════════════════════════════════
CAM_W, CAM_H = 640, 580
ROUND_DELAY = 5.0   # seconds between gesture lock-ins
PREP_SECONDS = 3.0  # time to position hand before round 1 timer starts


def game_screen():
    player_score = 0
    computer_score = 0
    ties = 0
    round_num = 0
    max_rounds = 5
    last_round_time = time.time()
    gesture = "---"
    computer_choice = "---"
    result = "Show your hand!"
    clock = pygame.time.Clock()

    countdown_screen()

    session_t0 = time.time()
    prep_until = session_t0 + PREP_SECONDS
    round_timer_armed = False

    while True:
        screen.fill(BG)
        update_shake()
        ox, oy = shake_offset

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cleanup(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cleanup(); sys.exit()

        ret, frame = cap.read()
        if not ret:
            break
        frame, gesture = process_frame(frame)
        cam_surf = frame_to_surface(frame)
        cam_surf = pygame.transform.smoothscale(cam_surf, (CAM_W, CAM_H))
        screen.blit(cam_surf, (10 + ox, 20 + oy))

        now = time.time()
        in_prep = now < prep_until
        if in_prep:
            prep_left = prep_until - now
        else:
            if not round_timer_armed:
                last_round_time = now
                round_timer_armed = True

        elapsed = now - last_round_time if round_timer_armed else 0.0
        lock_in_left = max(0.0, ROUND_DELAY - elapsed)
        lock_in_digit = int(math.ceil(lock_in_left - 1e-9)) if lock_in_left > 0 else 0

        if (
            not in_prep
            and round_timer_armed
            and elapsed >= ROUND_DELAY
            and gesture != "---"
            and round_num < max_rounds
        ):
            computer_choice = random.choice(["Rock", "Paper", "Scissors"])
            result = get_winner(gesture, computer_choice)
            round_num += 1

            if result == "You win!":
                player_score += 1
                trigger_flash(GREEN)
                spawn_particles(WIDTH // 2, HEIGHT // 2, GREEN, 35)
                play_snd(snd_win)
            elif result == "Computer wins!":
                computer_score += 1
                trigger_flash(RED)
                trigger_shake()
                play_snd(snd_lose)
            else:
                ties += 1
                trigger_flash(YELLOW)
                play_snd(snd_tie)

            last_round_time = now

            if round_num >= max_rounds:
                time.sleep(1.0)
                return player_score, computer_score

        # ── RIGHT PANEL ─────────────────────────────────────
        px = 665
        draw_rounded_rect(screen, PANEL, (px, 20, 420, 580), radius=22)
        accent_bar = pygame.Surface((420, 3), pygame.SRCALPHA)
        accent_bar.fill((*GLOW, 100))
        screen.blit(accent_bar, (px, 20))

        draw_text_centered(screen, f"Round  {min(round_num + 1, max_rounds)} / {max_rounds}",
                           font_tiny, GRAY, px + 210, 42)
        pygame.draw.line(screen, DARK_GRAY, (px + 24, 60), (px + 396, 60), 1)

        for i in range(max_rounds):
            cx = px + 58 + i * 76
            done = i < round_num
            color = GREEN if done else DARK_GRAY
            pygame.draw.circle(screen, color, (cx, 80), 9 if done else 6)

        pygame.draw.line(screen, DARK_GRAY, (px + 24, 100), (px + 396, 100), 1)

        draw_text(screen, "Your gesture", font_tiny, GRAY, px + 24, 110)
        g_color = GESTURE_COLOR.get(gesture, GRAY)
        draw_rounded_rect(screen, g_color, (px + 24, 130, 372, 54), radius=12, alpha=45)
        pygame.draw.rect(screen, g_color, (px + 24, 130, 4, 54), border_radius=2)
        draw_text_centered(screen, gesture, font_med, g_color, px + 210, 157)

        draw_text(screen, "Computer", font_tiny, GRAY, px + 24, 196)
        c_color = GESTURE_COLOR.get(computer_choice, GRAY)
        draw_rounded_rect(screen, c_color, (px + 24, 214, 372, 54), radius=12, alpha=45)
        pygame.draw.rect(screen, c_color, (px + 24, 214, 4, 54), border_radius=2)
        draw_text_centered(screen, computer_choice, font_med, c_color, px + 210, 241)

        if in_prep:
            draw_rounded_rect(screen, PURPLE, (px + 24, 268, 372, 88), radius=16, alpha=55)
            pygame.draw.rect(screen, PURPLE, (px + 24, 268, 4, 88), border_radius=2)
            draw_text_centered(screen, "Prepare gesture", font_small, PURPLE, px + 210, 292)
            draw_text_centered(screen, str(int(math.ceil(prep_left))), font_huge, WHITE,
                               px + 210, 332)
            draw_text_centered(screen, "Round 1 timer starts after", font_tiny, GRAY,
                               px + 210, 378)
            draw_rounded_rect(screen, DARK_GRAY, (px + 24, 392, 372, 36), radius=10, alpha=200)
            draw_text_centered(screen, "Hold position in the box", font_tiny, GRAY,
                               px + 210, 410)
        else:
            r_color = RESULT_COLOR.get(result, GRAY)
            draw_rounded_rect(screen, r_color, (px + 24, 280, 372, 58), radius=14, alpha=48)
            draw_text_centered(screen, result, font_med, r_color, px + 210, 309)

        if not in_prep:
            draw_text(screen, "Lock in", font_tiny, GRAY, px + 24, 352)
            bar_bg = (px + 24, 372, 372, 12)
            pygame.draw.rect(screen, DARK_GRAY, bar_bg, border_radius=6)
            prog = min(1.0, elapsed / ROUND_DELAY) if round_timer_armed else 0.0
            if prog > 0:
                pygame.draw.rect(screen, GLOW, (px + 24, 372, max(10, int(372 * prog)), 12),
                                 border_radius=6)
            cd_col = YELLOW if lock_in_digit <= 2 else WHITE
            draw_text_centered(screen, f"{lock_in_digit}s", font_big, cd_col, px + 210, 404)
            draw_text_centered(screen, f"next read in {lock_in_digit}s" if lock_in_digit else "show gesture!",
                               font_tiny, GRAY, px + 210, 448)

        if in_prep:
            ban = pygame.Surface((CAM_W, 76), pygame.SRCALPHA)
            ban.fill((*PURPLE, 140))
            screen.blit(ban, (10 + ox, 20 + oy + CAM_H - 76))
            draw_text_centered(screen, f"PREPARE  •  {int(math.ceil(prep_left))}s",
                                font_med, WHITE, 10 + ox + CAM_W // 2, 20 + oy + CAM_H - 42)

        pygame.draw.line(screen, DARK_GRAY, (px + 24, 468), (px + 396, 468), 1)
        draw_text_centered(screen, "Score", font_small, WHITE, px + 210, 488)

        draw_rounded_rect(screen, GREEN, (px + 24, 504, 108, 52), radius=12, alpha=38)
        draw_text_centered(screen, "You", font_tiny, GRAY, px + 78, 518)
        draw_text_centered(screen, str(player_score), font_med, GREEN, px + 78, 540)

        draw_rounded_rect(screen, YELLOW, (px + 152, 504, 108, 52), radius=12, alpha=38)
        draw_text_centered(screen, "Tie", font_tiny, GRAY, px + 206, 518)
        draw_text_centered(screen, str(ties), font_med, YELLOW, px + 206, 540)

        draw_rounded_rect(screen, RED, (px + 280, 504, 108, 52), radius=12, alpha=38)
        draw_text_centered(screen, "PC", font_tiny, GRAY, px + 334, 518)
        draw_text_centered(screen, str(computer_score), font_med, RED, px + 334, 540)

        draw_text_centered(screen, "Q — quit", font_tiny, DARK_GRAY, px + 210, 574)

        update_draw_particles()
        draw_flash()

        pygame.display.flip()
        clock.tick(60)

    return player_score, computer_score


# ════════════════════════════════════════════════════════════
#  CLEANUP & MAIN
# ════════════════════════════════════════════════════════════
def cleanup():
    cap.release()
    pygame.quit()

start_screen()

while True:
    ps, cs = game_screen()
    play_again = winner_screen(ps, cs)
    if not play_again:
        break
    gesture_history.clear()
    particles.clear()

cleanup()
