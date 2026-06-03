"""Exporta el modelo SigLIP2 EXISTENTE a ONNX para correr on-device.

NO entrena nada: descarga/usa el modelo ya publicado
`prithivMLmods/Alphabet-Sign-Language-Detection` (el mismo que usa el
proyecto Python con webcam) y lo exporta a ONNX. Luego `quantize_onnx.py`
lo cuantiza (int8) y lo publica como asset de la app Flutter, donde se
ejecuta con ONNX Runtime sin ningun servidor.

Pipeline:  transformers/PyTorch  ->  ONNX  ->  (quantize_onnx.py) int8

Entrada del modelo : 1x3x224x224 (NCHW)
Salida             : logits de 26 clases (A-Z)

Uso:
    py -3.10 export_siglip_onnx.py
    py -3.10 quantize_onnx.py
"""
from pathlib import Path

import torch
from transformers import SiglipForImageClassification

HERE = Path(__file__).resolve().parent
WORK = HERE / "build"

HF_REPO_ID = "prithivMLmods/Alphabet-Sign-Language-Detection"
ONNX_PATH = WORK / "siglip_asl.onnx"


class LogitsOnly(torch.nn.Module):
    """Envuelve el modelo para que el forward devuelva solo el tensor logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    print(f"[+] Cargando modelo {HF_REPO_ID} (se descarga la 1a vez ~372 MB)...")
    model = SiglipForImageClassification.from_pretrained(HF_REPO_ID)
    model.eval()

    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    print(f"[+] Clases ({len(labels)}): {labels}")

    WORK.mkdir(parents=True, exist_ok=True)
    print("[+] Exportando a ONNX...")
    torch.onnx.export(
        LogitsOnly(model),
        torch.randn(1, 3, 224, 224),
        ONNX_PATH.as_posix(),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] ONNX guardado en {ONNX_PATH} ({mb:.1f} MB)")
    print("     Ahora ejecuta: py -3.10 quantize_onnx.py")


if __name__ == "__main__":
    main()
