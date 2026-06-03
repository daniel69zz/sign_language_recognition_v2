# ASL Sign Language Detector

Detección en vivo del alfabeto del lenguaje de señas americano (ASL, A-Z)
usando webcam. Combina **MediaPipe Hands** para localizar la mano y un
modelo **SigLIP2** (`prithivMLmods/Alphabet-Sign-Language-Detection`)
fine-tuneado para clasificar cada letra.

Incluye un dashboard con sidebar, brackets sobre la mano detectada,
constructor de palabras por estabilidad temporal y modo espejo.

---

## Servidor REST (para la app Flutter)

Además de la app de webcam, este repo incluye `flask_server.py`: el mismo
pipeline (MediaPipe + SigLIP2) expuesto como API REST que consume la app
Flutter de `../HeHa`.

### Con Docker (recomendado)

```bash
docker compose up --build        # http://0.0.0.0:5000
```

Los modelos se descargan una vez y se persisten en el volumen `asl-models`.

### Sin Docker

```bash
pip install -r requirements_server.txt
python flask_server.py
```

Endpoints: `GET /health`, `POST /recognize` (multipart `image`), `GET /info`.

---

## Requisitos (app de webcam)

- **Python 3.12.1** (probado en esta versión; debería funcionar en 3.10–3.12)
- Webcam conectada
- Conexión a internet en el primer arranque (se descargan los modelos)
- Opcional: GPU con CUDA para inferencia más rápida (cae a CPU automáticamente)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd PROYECTO_MOVILES_SIGNS_V3
```

### 2. Crear y activar entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Si querés usar GPU con CUDA, instalá `torch` siguiendo las instrucciones
> oficiales de https://pytorch.org/get-started/locally/ antes de
> `pip install -r requirements.txt`.

---

## Ejecución

```bash
python asl_webcam.py
```

En el primer arranque se descargan automáticamente:
- El modelo SigLIP2 desde Hugging Face (~370 MB).
- `hand_landmarker.task` de MediaPipe a `./models/`.

### Atajos de teclado

| Tecla       | Acción                  |
|-------------|-------------------------|
| `q`         | Salir                   |
| `c`         | Limpiar palabra         |
| `space`     | Añadir espacio          |
| `backspace` | Borrar última letra     |

---

## Configuración

Las constantes principales están al inicio de `asl_webcam.py`:

| Constante           | Default       | Descripción                                            |
|---------------------|---------------|--------------------------------------------------------|
| `CAMERA_INDEX`      | `1`           | Índice de la webcam (cambiá a `0` si solo tenés una).  |
| `CAM_W`, `CAM_H`    | `1280, 720`   | Resolución de captura.                                 |
| `CONF_THRESHOLD`    | `0.55`        | Confianza mínima para aceptar una letra.               |
| `STABILITY_FRAMES`  | `15`          | Frames seguidos con la misma letra para fijarla.       |
| `COOLDOWN_FRAMES`   | `25`          | Espera antes de poder fijar otra letra.                |
| `PREDICT_EVERY`     | `2`           | Cada cuántos frames correr el clasificador.            |

---

## Estructura del proyecto

```
PROYECTO_MOVILES_SIGNS_V3/
├── asl_webcam.py          # App principal
├── requirements.txt       # Dependencias Python
├── .python-version        # Versión recomendada de Python
├── .gitignore
├── README.md
└── models/                # Se crea al ejecutar; ignorado por git
    └── hand_landmarker.task
```

---

## Solución de problemas

- **No se abre la cámara:** cambiá `CAMERA_INDEX` a `0` en `asl_webcam.py`.
- **Error al descargar el modelo:** verificá conexión a internet y reintentá.
- **FPS bajos en CPU:** subí `PREDICT_EVERY` a `3` o `4`, o usá GPU.
- **`mediapipe` no instala:** asegurate de usar Python 3.10–3.12 (no 3.13+).
