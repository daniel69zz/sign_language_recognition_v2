# Servidor REST de reconocimiento ASL (MediaPipe + SigLIP2).
# Imagen CPU; los modelos se descargan en el primer arranque y se persisten
# en el volumen montado en /app/models.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf-cache

WORKDIR /app

# Dependencias de sistema para OpenCV / MediaPipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU primero, desde el indice oficial, para evitar el build CUDA (~2 GB).
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements_server.txt .
RUN pip install -r requirements_server.txt

COPY flask_server.py .

EXPOSE 5000

CMD ["python", "flask_server.py"]
