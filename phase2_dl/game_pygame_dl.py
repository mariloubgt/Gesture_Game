"""
Phase 2 — Rock Paper Scissors game using a trained MobileNetV2 CNN classifier.
Uses TFLite for fast CPU inference + background thread so the game never stalls.
"""

import cv2
import pygame
import random
import time
import numpy as np
import sys
import os
import math
import threading
from collections import deque, Counter

# ── TFLITE INFERENCE ENGINE ─────────────────────────────────
# Only import the lightweight TFLite runtime — no full TensorFlow overhead
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "model", "rps_mobilenet.tflite")

print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=TFLITE_PATH, num_threads=4)
interpreter.allocate_tensors()
_input_detail  = interpreter.get_input_details()[0]
_output_detail = interpreter.get_output_details()[0]
_input_idx     = _input_detail["index"]
_output_idx    = _output_detail["index"]

# Order MUST match train.py / dataset: paper=0, rock=1, scissors=2
CLASS_NAMES = ["Paper", "Rock", "Scissors"]
IDX_PAPER, IDX_ROCK, IDX_SCISSORS = 0, 1, 2
IMG_SIZE    = (224, 224)
# Paper softmax is often flatter than rock/scissors; use lower accept bar.
_MIN_CONF = {"Paper": 0.26, "Rock": 0.40, "Scissors": 0.40}
# Training crops are mostly “hand fills frame”. Zoom central ROI so webcam hands aren’t tiny at 224².
_ZOOM_WIDE = 0.82   # mild crop — keeps full gesture context
_ZOOM_TIGHT = 0.55  # stronger zoom — helps open palm when hand doesn’t fill the box
print("Model ready.")

# ── THREADED INFERENCE ──────────────────────────────────────
_lock        = threading.Lock()
_latest_pred = ("---", 0.0)
_infer_frame = None
_infer_ready = threading.Event()
_running     = True

def _center_crop(roi_bgr, frac):
    """Keep centre `frac` of width/height (0<frac<=1). frac=0.6 => digital zoom ~1.67×."""
    h, w = roi_bgr.shape[:2]
    frac = max(0.35, min(1.0, float(frac)))
    nw, nh = int(w * frac), int(h * frac)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    return roi_bgr[y0:y0 + nh, x0:x0 + nw]


def _preprocess(roi_bgr, zoom_frac=_ZOOM_WIDE):
    """Centre-zoom (match training scale), CLAHE, resize, MobileNet [-1,1] input."""
    cropped = _center_crop(roi_bgr, zoom_frac)
    if cropped.size == 0:
        cropped = roi_bgr
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    cropped = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
    resized = (resized / 127.5) - 1.0
    return np.expand_dims(resized, axis=0)


def _fuse_probs_for_paper(p_wide, p_zoom):
    """
    Blend two softmax views. When the tight zoom raises paper mass (small hand / far away),
    trust it more — that view is closer to dataset framing.
    """
    pw = np.asarray(p_wide, dtype=np.float64)
    pz = np.asarray(p_zoom, dtype=np.float64)
    if pz[IDX_PAPER] > pw[IDX_PAPER] + 0.05 and pz[IDX_PAPER] >= 0.18:
        return 0.28 * pw + 0.72 * pz
    return 0.62 * pw + 0.38 * pz


def _decode_gesture(probs):
    """
    Map softmax to label. Open palm (paper) is often a close second to rock;
    use margin rules so paper is not dropped when argmax is rock with low gap.
    """
    p = np.asarray(probs, dtype=np.float64)
    p_paper = float(p[IDX_PAPER])
    p_rock = float(p[IDX_ROCK])
    p_sci = float(p[IDX_SCISSORS])
    top_i = int(np.argmax(p))
    top_p = float(p[top_i])

    # Rock vs paper: model frequently picks rock on real webcam palms / small hands
    if top_i == IDX_ROCK:
        if p_paper >= 0.18 and (p_rock - p_paper) <= 0.28:
            label, conf = "Paper", p_paper
        else:
            label, conf = "Rock", p_rock
    elif top_i == IDX_SCISSORS:
        label, conf = "Scissors", p_sci
    else:
        label, conf = "Paper", p_paper

    if conf < _MIN_CONF[label]:
        return "---", conf
    return label, conf


def _inference_worker():
    """Background thread: picks up frames and runs TFLite inference."""
    global _latest_pred
    while _running:
        _infer_ready.wait(timeout=0.5)
        _infer_ready.clear()
        with _lock:
            frame = _infer_frame
        if frame is None:
            continue
        inp_w = _preprocess(frame, _ZOOM_WIDE)
        interpreter.set_tensor(_input_idx, inp_w)
        interpreter.invoke()
        p_wide = interpreter.get_tensor(_output_idx)[0].astype(np.float64)

        inp_z = _preprocess(frame, _ZOOM_TIGHT)
        interpreter.set_tensor(_input_idx, inp_z)
        interpreter.invoke()
        p_zoom = interpreter.get_tensor(_output_idx)[0].astype(np.float64)

        preds = _fuse_probs_for_paper(p_wide, p_zoom)
        label, conf = _decode_gesture(preds)
        with _lock:
            _latest_pred = (label, conf)

_worker = threading.Thread(target=_inference_worker, daemon=True)
_worker.start()

def submit_roi(roi_bgr):
    """Hand an ROI to the background thread (non-blocking)."""
    global _infer_frame
    with _lock:
        _infer_frame = roi_bgr.copy()
    _infer_ready.set()

def get_prediction():
    with _lock:
        return _latest_pred

# ── INIT ────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

WIDTH, HEIGHT = 1100, 620
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors — Deep Learning")

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

# ── ROI (slightly tighter than before: hand fills more pixels at normal arm length) ──
ROI_X_FRAC = 0.30
ROI_Y_FRAC = 0.10
ROI_W_FRAC = 0.40
ROI_H_FRAC = 0.68

def roi_rect(fh, fw):
    x = int(fw * ROI_X_FRAC)
    y = int(fh * ROI_Y_FRAC)
    w = int(fw * ROI_W_FRAC)
    h = int(fh * ROI_H_FRAC)
    return x, y, w, h

# ── GESTURE HISTORY ────────────────────────────────────────
gesture_history = deque(maxlen=15)
conf_history    = deque(maxlen=15)
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

def stable_gesture(history, confs):
    """Confidence-weighted vote; slight boost for Paper so open palm stabilises faster."""
    window = list(history)[-9:]
    cwindow = list(confs)[-9:]
    paper_boost = 1.28

    scores = {}
    for g, c in zip(window, cwindow):
        if g == "---":
            continue
        w = paper_boost if g == "Paper" else 1.0
        scores[g] = scores.get(g, 0.0) + c * w

    if not scores:
        return "---"

    best = max(scores, key=scores.get)
    count = sum(1 for g in window if g == best)
    if count >= 2:
        return best
    return "---"

# ── PROCESS FRAME ──────────────────────────────────────────
_frame_count = 0
# Two TFLite runs per submit (wide + tight zoom); skip more frames to keep CPU similar.
INFER_EVERY  = 3

def process_frame(frame):
    global _frame_count
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rx, ry, rw, rh = roi_rect(fh, fw)
    roi = frame[ry:ry + rh, rx:rx + rw]

    _frame_count += 1
    if roi.size > 0 and _frame_count % INFER_EVERY == 0:
        submit_roi(roi)

    detected, conf = get_prediction()
    # Thresholds applied inside _decode_gesture (per-class)

    gesture_history.append(detected)
    conf_history.append(conf)
    stable = stable_gesture(gesture_history, conf_history)

    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (120, 220, 255), 2)
    label = f"{stable}  ({conf:.0%})" if conf > 0 else stable
    cv2.putText(frame, label, (rx, max(20, ry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 255), 2, cv2.LINE_AA)

    return frame, stable

# ── FRAME → PYGAME SURFACE ─────────────────────────────────
def frame_to_surface(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB").copy()

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
        draw_text_centered(screen, "Deep Learning (MobileNetV2)", font_med, WHITE, WIDTH // 2, 218)
        draw_text_centered(screen, "Center your hand in the box — open palm works best when the hand fills most of it.",
                           font_small, GRAY, WIDTH // 2, 262)
        draw_text_centered(screen, "3s prepare after GO! — then 5s each round to lock your gesture.",
                           font_tiny, GRAY, WIDTH // 2, 292)

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
            pygame.draw.rect(screen, DARK_GRAY, (px + 24, 372, 372, 12), border_radius=6)
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
    global _running
    _running = False
    cap.release()
    pygame.quit()

start_screen()

while True:
    ps, cs = game_screen()
    play_again = winner_screen(ps, cs)
    if not play_again:
        break
    gesture_history.clear()
    conf_history.clear()
    particles.clear()

cleanup()
