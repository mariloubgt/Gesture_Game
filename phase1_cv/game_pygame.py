import cv2
import pygame
import random
import time
import numpy as np
import sys
import math
import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
from collections import deque, Counter

# ── MEDIAPIPE HAND LANDMARKER (new Tasks API) ───────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

_latest_result = None

def _result_callback(result, image, timestamp_ms):
    global _latest_result
    _latest_result = result

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.45,
    result_callback=_result_callback,
)
landmarker = HandLandmarker.create_from_options(hand_options)

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

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

# ── STATE ───────────────────────────────────────────────────
gesture_history = deque(maxlen=12)
_flash_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
_frame_ts = 0

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

# ── FINGER STATE DETECTION ─────────────────────────────────
def is_finger_extended(lm, tip_id, pip_id, mcp_id):
    """
    Check if a finger is extended using both y-comparison and
    distance-from-wrist ratio, so it works even when the hand is rotated.
    """
    tip = lm[tip_id]
    pip_pt = lm[pip_id]
    mcp = lm[mcp_id]
    wrist = lm[WRIST]

    # Method 1: tip is above PIP (works for upright hand)
    y_extended = tip.y < pip_pt.y

    # Method 2: tip is farther from wrist than PIP (works at any rotation)
    tip_dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    pip_dist = math.hypot(pip_pt.x - wrist.x, pip_pt.y - wrist.y)
    dist_extended = tip_dist > pip_dist * 1.05

    # Method 3: tip-to-mcp distance vs pip-to-mcp (finger straightness)
    tip_mcp = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
    pip_mcp = math.hypot(pip_pt.x - mcp.x, pip_pt.y - mcp.y)
    straight_extended = tip_mcp > pip_mcp * 1.2

    # Count votes — extended if at least 2 of 3 agree
    votes = int(y_extended) + int(dist_extended) + int(straight_extended)
    return votes >= 2

def get_finger_states(lm):
    """Returns dict of which fingers are extended."""
    # Thumb uses a different test
    palm_cx = (lm[WRIST].x + lm[INDEX_MCP].x) / 2
    palm_cy = (lm[WRIST].y + lm[INDEX_MCP].y) / 2
    tip_dist = math.hypot(lm[THUMB_TIP].x - palm_cx, lm[THUMB_TIP].y - palm_cy)
    ip_dist = math.hypot(lm[THUMB_IP].x - palm_cx, lm[THUMB_IP].y - palm_cy)
    thumb_ext = tip_dist > ip_dist * 1.2

    index_ext = is_finger_extended(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_ext = is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_ext = is_finger_extended(lm, RING_TIP, RING_PIP, RING_MCP)
    pinky_ext = is_finger_extended(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

    return {
        "thumb": thumb_ext,
        "index": index_ext,
        "middle": middle_ext,
        "ring": ring_ext,
        "pinky": pinky_ext,
    }

def classify_gesture(lm):
    """Classify Rock / Paper / Scissors from landmark positions."""
    fingers = get_finger_states(lm)
    index = fingers["index"]
    middle = fingers["middle"]
    ring = fingers["ring"]
    pinky = fingers["pinky"]

    main_up = sum([index, middle, ring, pinky])

    # Scissors: index + middle up, ring + pinky down (thumb ignored)
    if index and middle and not ring and not pinky:
        return "Scissors", main_up

    # Paper: 3-4 of the main fingers extended
    if main_up >= 3:
        return "Paper", main_up

    # Rock: 0-1 main fingers extended
    if main_up <= 1:
        return "Rock", main_up

    # 2 fingers but not index+middle → likely a sloppy scissors
    if main_up == 2 and (index or middle):
        return "Scissors", main_up

    return "Rock", main_up

def stable_gesture(history):
    real = [g for g in history if g != "---"]
    if len(real) < 3:
        return "---"
    recent = list(history)[-7:]
    real_recent = [g for g in recent if g != "---"]
    if not real_recent:
        return "---"
    best, cnt = Counter(real_recent).most_common(1)[0]
    if cnt >= 3:
        return best
    return "---"

# ── DRAW HAND ON FRAME ─────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def draw_hand_on_frame(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 220, 255), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = (60, 255, 160) if i in (4, 8, 12, 16, 20) else (255, 200, 80)
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)

# ── PROCESS FRAME ──────────────────────────────────────────
def process_frame(frame):
    global _latest_result, _frame_ts

    # Mirror the frame so it feels like a mirror to the user
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    _frame_ts += 33
    landmarker.detect_async(mp_image, _frame_ts)

    detected = "---"
    n_fingers = -1
    result = _latest_result

    if result and result.hand_landmarks:
        lm_list = result.hand_landmarks[0]
        draw_hand_on_frame(frame, lm_list)
        detected, n_fingers = classify_gesture(lm_list)

    gesture_history.append(detected)
    stable = stable_gesture(gesture_history)

    label = stable
    if n_fingers >= 0:
        label += f"  ({n_fingers})"
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2, cv2.LINE_AA)

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
        draw_text_centered(screen, "Gesture Recognition", font_med, WHITE, WIDTH // 2, 218)
        draw_text_centered(screen, "Hold your hand in front of the camera — no keyboard needed.",
                           font_small, GRAY, WIDTH // 2, 272)

        labels = ["Rock", "Paper", "Scissors"]
        colors = [ORANGE, BLUE, ACCENT]
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            cx = WIDTH // 2 - 130 + i * 130
            cy = int(338 + 5 * math.sin(pulse + i * 1.1))
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
#  COUNTDOWN
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
            draw_text_centered(screen, label, font_huge, color, WIDTH // 2, HEIGHT // 2)

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

def game_screen():
    player_score = 0
    computer_score = 0
    ties = 0
    round_num = 0
    max_rounds = 5
    last_round_time = time.time()
    round_delay = 3.0
    gesture = "---"
    computer_choice = "---"
    result = "Show your hand!"
    clock = pygame.time.Clock()

    countdown_screen()

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
        elapsed = now - last_round_time
        countdown = max(0, int(round_delay - elapsed))

        if elapsed >= round_delay and gesture != "---" and round_num < max_rounds:
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

        r_color = RESULT_COLOR.get(result, GRAY)
        draw_rounded_rect(screen, r_color, (px + 24, 280, 372, 58), radius=14, alpha=48)
        draw_text_centered(screen, result, font_med, r_color, px + 210, 309)

        draw_text(screen, "Next round", font_tiny, GRAY, px + 24, 352)
        for i in range(3):
            cx = px + 68 + i * 118
            active = (3 - i) <= countdown
            dot_col = YELLOW if active else DARK_GRAY
            draw_rounded_rect(screen, dot_col, (cx - 26, 368, 52, 40), radius=10)
            draw_text_centered(screen, str(3 - i), font_med,
                               BG if active else GRAY, cx, 388)

        pygame.draw.line(screen, DARK_GRAY, (px + 24, 428), (px + 396, 428), 1)
        draw_text_centered(screen, "Score", font_small, WHITE, px + 210, 448)

        draw_rounded_rect(screen, GREEN, (px + 24, 468, 108, 60), radius=12, alpha=38)
        draw_text_centered(screen, "You", font_tiny, GRAY, px + 78, 484)
        draw_text_centered(screen, str(player_score), font_med, GREEN, px + 78, 510)

        draw_rounded_rect(screen, YELLOW, (px + 152, 468, 108, 60), radius=12, alpha=38)
        draw_text_centered(screen, "Tie", font_tiny, GRAY, px + 206, 484)
        draw_text_centered(screen, str(ties), font_med, YELLOW, px + 206, 510)

        draw_rounded_rect(screen, RED, (px + 280, 468, 108, 60), radius=12, alpha=38)
        draw_text_centered(screen, "PC", font_tiny, GRAY, px + 334, 484)
        draw_text_centered(screen, str(computer_score), font_med, RED, px + 334, 510)

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
    landmarker.close()
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
