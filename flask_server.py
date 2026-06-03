"""Servidor Flask para reconocimiento ASL (MediaPipe + SigLIP2).

Replica el pipeline de `asl_webcam.py` pero como API REST para la app Flutter:
recibe una imagen, detecta la mano con MediaPipe, la recorta y la clasifica
con el modelo SigLIP2 fine-tuneado. Devuelve la letra, la confianza y si se
detecto una mano.

Endpoints:
    GET  /health     -> estado del servidor y si los modelos estan cargados
    POST /recognize  -> multipart con campo 'image' -> {letter, confidence, ...}
    GET  /info       -> metadatos del servidor

Ejecucion:
    pip install -r requirements_server.txt
    python flask_server.py
El servidor queda en http://0.0.0.0:5000 (accesible desde el celular en la
misma red local).
"""
import io

import cv2
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, jsonify, request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# --- Config -----------------------------------------------------------------
HF_REPO_ID = "prithivMLmods/Alphabet-Sign-Language-Detection"
HAND_TASK_FILENAME = "hand_landmarker.task"
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
CONF_THRESHOLD = 0.55
HAND_PAD = 40

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
PROCESSOR = None
ID2LABEL = None
HAND_DETECTOR = None


# --- Carga de modelos -------------------------------------------------------
def _download_hand_task() -> str:
    from pathlib import Path
    import urllib.request

    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(exist_ok=True)
    target = cache_dir / HAND_TASK_FILENAME
    if not target.exists():
        print(f"[+] Descargando {HAND_TASK_FILENAME}...")
        urllib.request.urlretrieve(HAND_TASK_URL, target)
    return str(target)


def load_models():
    global MODEL, PROCESSOR, ID2LABEL, HAND_DETECTOR
    print(f"[+] Cargando {HF_REPO_ID} en {DEVICE}...")
    PROCESSOR = AutoImageProcessor.from_pretrained(HF_REPO_ID)
    MODEL = SiglipForImageClassification.from_pretrained(HF_REPO_ID)
    MODEL.to(DEVICE)
    MODEL.eval()
    ID2LABEL = MODEL.config.id2label

    task_path = _download_hand_task()
    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    HAND_DETECTOR = mp_vision.HandLandmarker.create_from_options(options)
    print("[+] Modelos cargados.")


# --- Inferencia -------------------------------------------------------------
def hand_bbox(landmarks, w, h, pad=HAND_PAD):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = side // 2
    return (
        max(0, cx - half),
        max(0, cy - half),
        min(w, cx + half),
        min(h, cy + half),
    )


def classify(crop_bgr):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inputs = PROCESSOR(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(MODEL(**inputs).logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
    return ID2LABEL[int(top_idx.item())], float(top_prob.item())


# --- Endpoints --------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify(
        status="ok",
        models_loaded=MODEL is not None and HAND_DETECTOR is not None,
        device=str(DEVICE),
    )


@app.get("/info")
def info():
    letters = list(ID2LABEL.values()) if ID2LABEL else []
    return jsonify(
        name="ASL Recognition Server",
        version="3.0.0",
        supported_letters=letters,
        device=str(DEVICE),
    )


@app.post("/recognize")
def recognize():
    if "image" not in request.files:
        return jsonify(success=False, error="falta el campo 'image'"), 400

    file_bytes = request.files["image"].read()
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    frame_rgb = np.array(pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = HAND_DETECTOR.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify(
            success=True, letter="--", confidence=0.0, hand_detected=False
        )

    bbox = hand_bbox(result.hand_landmarks[0], w, h)
    if bbox is None:
        return jsonify(
            success=True, letter="--", confidence=0.0, hand_detected=False
        )

    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return jsonify(
            success=True, letter="--", confidence=0.0, hand_detected=False
        )

    letter, confidence = classify(crop)
    accepted = confidence >= CONF_THRESHOLD
    return jsonify(
        success=True,
        letter=letter if accepted else "--",
        confidence=confidence,
        hand_detected=True,
    )


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, threaded=True)
