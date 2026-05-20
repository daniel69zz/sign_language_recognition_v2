"""Detección de lenguaje de señas ASL en vivo con webcam.

Modelo: prithivMLmods/Alphabet-Sign-Language-Detection
(SigLIP2 base patch16-224 fine-tuned para letras A-Z).
Se usa MediaPipe Hands para recortar la mano antes de clasificar.

UI tipo dashboard con sidebar, brackets en la mano detectada,
constructor de palabras por estabilidad temporal y modo espejo.

Instalación de dependencias:
    pip install torch torchvision transformers mediapipe opencv-python pillow huggingface_hub
"""
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# --- Config -----------------------------------------------------------------
HF_REPO_ID = "prithivMLmods/Alphabet-Sign-Language-Detection"
# Hand landmarker oficial de Google (se descarga vía HF)
HAND_TASK_REPO = "qualcomm/MediaPipe-Hand-Detection"
HAND_TASK_FILENAME = "hand_landmarker.task"
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CONF_THRESHOLD = 0.55
CAMERA_INDEX = 1
CAM_W, CAM_H = 1280, 720
SIDEBAR_W = 400
WINDOW_W, WINDOW_H = 1500, 640

# Padding alrededor de los landmarks de la mano (px)
HAND_PAD = 40

# Frames seguidos con la misma letra para "fijarla" en la palabra
STABILITY_FRAMES = 15
# Frames de espera tras fijar antes de poder fijar de nuevo
COOLDOWN_FRAMES = 25

# Procesar el clasificador cada N frames (más rápido)
PREDICT_EVERY = 2

# Paleta (BGR)
COL_BG      = (28, 26, 34)
COL_PANEL   = (44, 42, 54)
COL_PANEL2  = (62, 58, 76)
COL_TEXT    = (240, 240, 245)
COL_MUTED   = (150, 145, 165)
COL_ACCENT  = (180, 220, 90)
COL_ACCENT2 = (220, 130, 230)
COL_OK      = (90, 220, 140)
COL_WARN    = (90, 160, 230)


# --- Modelo -----------------------------------------------------------------
def load_model(device: torch.device):
    print(f"[+] Descargando/cargando {HF_REPO_ID}...")
    processor = AutoImageProcessor.from_pretrained(HF_REPO_ID)
    model = SiglipForImageClassification.from_pretrained(HF_REPO_ID)
    model.to(device)
    model.eval()
    id2label = model.config.id2label
    print(f"[+] Clases ({len(id2label)}): {list(id2label.values())}")
    print(f"[+] Dispositivo: {device}")
    return model, processor, id2label


def _download_hand_task() -> str:
    """Descarga el modelo hand_landmarker.task a ./models/."""
    from pathlib import Path
    import urllib.request

    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(exist_ok=True)
    target = cache_dir / HAND_TASK_FILENAME
    if not target.exists():
        print(f"[+] Descargando {HAND_TASK_FILENAME} desde Google...")
        urllib.request.urlretrieve(HAND_TASK_URL, target)
    return str(target)


def make_hand_detector():
    task_path = _download_hand_task()
    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# --- Helpers de dibujo ------------------------------------------------------
def draw_corner_box(img, x1, y1, x2, y2, color, thickness=3, length=22):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def put_text(img, text, org, scale=0.6, color=COL_TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_hand_box(frame, bbox, label, conf):
    x1, y1, x2, y2 = bbox
    color = COL_ACCENT if conf >= 0.6 else COL_WARN
    draw_corner_box(frame, x1, y1, x2, y2, color, thickness=3, length=24)

    text = f"{label}  {conf * 100:.0f}%"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    pad = 6
    tag_x2 = x1 + tw + 2 * pad
    if y1 - th - bl - 2 * pad < 0:
        tag_y1 = y1
        text_y = y1 + th + pad
    else:
        tag_y1 = y1 - th - bl - 2 * pad
        text_y = y1 - bl - pad
    cv2.rectangle(frame, (x1, tag_y1),
                  (tag_x2, tag_y1 + th + bl + 2 * pad), color, -1)
    put_text(frame, text, (x1 + pad, text_y),
             scale=0.7, color=(20, 20, 20))


# --- Sidebar ----------------------------------------------------------------
def make_sidebar(h, w, *, top_label, top_conf, conf_smooth,
                 fps, word, streak_pct):
    panel = np.full((h, w, 3), COL_BG, dtype=np.uint8)

    put_text(panel, "ASL DETECTOR", (24, 50),
             scale=1.0, color=COL_ACCENT, thickness=2)
    put_text(panel, "siglip2  +  mediapipe", (24, 76),
             scale=0.5, color=COL_MUTED)
    cv2.line(panel, (24, 95), (w - 24, 95), COL_PANEL, 2)

    # --- Card DETECTED ---
    card_y, card_h = 115, 240
    cv2.rectangle(panel, (20, card_y), (w - 20, card_y + card_h),
                  COL_PANEL, -1)
    put_text(panel, "DETECTED", (36, card_y + 30),
             scale=0.55, color=COL_MUTED)

    if top_label is not None:
        letter = str(top_label)
        (lw, lh), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_DUPLEX, 5.5, 8)
        cx = 20 + (w - 40) // 2 - lw // 2
        put_text(panel, letter, (cx, card_y + 165),
                 scale=5.5, color=COL_ACCENT, thickness=8)

        bar_x1, bar_x2 = 36, w - 36
        bar_y, bar_h = card_y + 195, 14
        cv2.rectangle(panel, (bar_x1, bar_y), (bar_x2, bar_y + bar_h),
                      COL_PANEL2, -1)
        fill = int((bar_x2 - bar_x1) * max(0.0, min(1.0, conf_smooth)))
        bar_color = COL_OK if conf_smooth >= 0.6 else COL_WARN
        cv2.rectangle(panel, (bar_x1, bar_y),
                      (bar_x1 + fill, bar_y + bar_h), bar_color, -1)
        put_text(panel, f"{conf_smooth * 100:.0f}% confidence",
                 (bar_x1, bar_y + bar_h + 22), scale=0.55, color=COL_TEXT)
    else:
        put_text(panel, "--", (w // 2 - 35, card_y + 160),
                 scale=4.5, color=COL_MUTED, thickness=6)
        put_text(panel, "muestra una sena a la camara",
                 (36, card_y + 210), scale=0.5, color=COL_MUTED)

    # --- Card WORD ---
    word_y, word_h = card_y + card_h + 25, 140
    cv2.rectangle(panel, (20, word_y), (w - 20, word_y + word_h),
                  COL_PANEL, -1)
    put_text(panel, "WORD", (36, word_y + 30),
             scale=0.55, color=COL_MUTED)

    shown = word[-14:] if len(word) > 14 else word
    put_text(panel, shown + "_", (36, word_y + 95),
             scale=1.7, color=COL_TEXT, thickness=2)

    ring_cx, ring_cy, ring_r = w - 60, word_y + 70, 24
    cv2.circle(panel, (ring_cx, ring_cy), ring_r, COL_PANEL2, 3)
    end_angle = int(360 * streak_pct)
    if end_angle > 0:
        cv2.ellipse(panel, (ring_cx, ring_cy), (ring_r, ring_r),
                    -90, 0, end_angle, COL_ACCENT2, 3)
    pct_txt = f"{int(streak_pct * 100)}%"
    (tw, _), _ = cv2.getTextSize(pct_txt, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    put_text(panel, pct_txt, (ring_cx - tw // 2, ring_cy + 5),
             scale=0.45, color=COL_MUTED)

    # --- Footer ---
    foot_y = h - 150
    cv2.line(panel, (24, foot_y), (w - 24, foot_y), COL_PANEL, 2)

    put_text(panel, f"FPS  {fps:5.1f}",
             (24, foot_y + 32), scale=0.65, color=COL_TEXT)

    hotkeys = [
        ("[q]", "salir"),
        ("[c]", "limpiar palabra"),
        ("[space]", "anadir espacio"),
        ("[backspace]", "borrar letra"),
    ]
    y = foot_y + 60
    for k, desc in hotkeys:
        put_text(panel, k, (24, y), scale=0.5, color=COL_ACCENT2)
        put_text(panel, desc, (130, y), scale=0.5, color=COL_MUTED)
        y += 22

    return panel


# --- Inferencia -------------------------------------------------------------
def hand_bbox_from_landmarks(landmarks, w, h, pad=HAND_PAD):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    # Bbox cuadrado (mejor para input 224x224)
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return x1, y1, x2, y2


def predict_letter(model, processor, id2label, device, frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
    label = id2label[int(top_idx.item())]
    return str(label), float(top_prob.item())


# --- Main loop --------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, id2label = load_model(device)
    hand_detector = make_hand_detector()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara (índice {CAMERA_INDEX}).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    window = "ASL Detector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, WINDOW_W, WINDOW_H)

    prev = time.time()
    fps_smooth = 0.0
    conf_smooth = 0.0

    word = ""
    last_letter = None
    streak = 0
    cooldown = 0
    frame_idx = 0

    cached_label, cached_conf = None, 0.0
    timestamp_ms = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            top_label, top_conf = None, 0.0
            bbox = None

            if hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                bbox = hand_bbox_from_landmarks(landmarks, w, h)
                if bbox is not None:
                    if frame_idx % PREDICT_EVERY == 0:
                        label, conf = predict_letter(
                            model, processor, id2label, device, frame, bbox,
                        )
                        cached_label, cached_conf = label, conf
                    else:
                        label, conf = cached_label, cached_conf

                    if label is not None and conf >= CONF_THRESHOLD:
                        top_label, top_conf = label, conf
                        draw_hand_box(frame, bbox, top_label, top_conf)
                    else:
                        draw_corner_box(frame, *bbox, COL_MUTED,
                                        thickness=2, length=20)
            else:
                cached_label, cached_conf = None, 0.0

            if top_label is None:
                last_letter = None
                streak = 0
            else:
                if top_label == last_letter:
                    streak += 1
                else:
                    last_letter = top_label
                    streak = 1

            if cooldown > 0:
                cooldown -= 1

            if (last_letter is not None
                    and streak >= STABILITY_FRAMES
                    and cooldown == 0):
                word += last_letter
                cooldown = COOLDOWN_FRAMES
                streak = 0
                last_letter = None

            conf_smooth = 0.7 * conf_smooth + 0.3 * top_conf
            streak_pct = min(1.0, streak / STABILITY_FRAMES)

            now = time.time()
            dt = now - prev
            prev = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = (0.9 * fps_smooth + 0.1 * inst_fps
                          if fps_smooth else inst_fps)

            sidebar = make_sidebar(
                frame.shape[0], SIDEBAR_W,
                top_label=top_label,
                top_conf=top_conf,
                conf_smooth=conf_smooth,
                fps=fps_smooth,
                word=word,
                streak_pct=streak_pct,
            )
            canvas = np.hstack([frame, sidebar])

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                word = ""
            elif key == ord(" "):
                word += " "
            elif key == 8:
                word = word[:-1]

            frame_idx += 1
    finally:
        cap.release()
        hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
