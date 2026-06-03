"""Cuantiza el ONNX exportado de SigLIP2 y verifica que sigue infiriendo.

Usa cuantizacion dinamica int8 (sin datos de calibracion): reduce el tamano
~4x sobre las capas lineales del transformer, manteniendo el mismo modelo.
El resultado se corre on-device con ONNX Runtime (paquete Flutter onnxruntime).

Genera:
    ../../HeHa/assets/model/asl_model.onnx   (cuantizado)
    ../../HeHa/assets/model/labels.txt

Uso:
    py -3.10 quantize_onnx.py
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: F401

HERE = Path(__file__).resolve().parent
WORK = HERE / "build"
OUT_DIR = HERE.parent.parent / "HeHa" / "assets" / "model"

FP32_ONNX = WORK / "siglip_asl.onnx"
QUANT_ONNX = WORK / "siglip_asl_int8.onnx"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def quantize():
    print("[+] Cuantizando ONNX (dynamic int8, solo MatMul)...")
    # Solo MatMul: evita ConvInteger (la conv del patch-embedding no esta
    # implementada en ORT) y cuantiza donde estan la mayoria de los pesos.
    quantize_dynamic(
        model_input=FP32_ONNX.as_posix(),
        model_output=QUANT_ONNX.as_posix(),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    mb = QUANT_ONNX.stat().st_size / (1024 * 1024)
    fp = FP32_ONNX.stat().st_size / (1024 * 1024)
    print(f"[+] fp32: {fp:.1f} MB  ->  int8: {mb:.1f} MB")


def verify():
    print("[+] Verificando inferencia con ONNX Runtime...")
    dummy = np.random.rand(1, 3, 224, 224).astype(np.float32) * 2 - 1

    def top1(path):
        sess = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        logits = sess.run(None, {in_name: dummy})[0][0]
        idx = int(np.argmax(logits))
        exp = np.exp(logits - logits.max())
        prob = float(exp[idx] / exp.sum())
        return in_name, LETTERS[idx], prob

    name_fp, letter_fp, prob_fp = top1(FP32_ONNX)
    name_q, letter_q, prob_q = top1(QUANT_ONNX)
    print(f"[+] input name: {name_fp}")
    print(f"[+] fp32 -> {letter_fp} ({prob_fp * 100:.1f}%)")
    print(f"[+] int8 -> {letter_q} ({prob_q * 100:.1f}%)")
    return name_q


def publish(input_name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(QUANT_ONNX, OUT_DIR / "asl_model.onnx")
    (OUT_DIR / "labels.txt").write_text("\n".join(LETTERS), encoding="utf-8")
    mb = (OUT_DIR / "asl_model.onnx").stat().st_size / (1024 * 1024)
    print(f"[+] Publicado {OUT_DIR / 'asl_model.onnx'} ({mb:.1f} MB)")
    print(f"[+] Nombre de entrada del modelo: {input_name}")


def main():
    quantize()
    name = verify()
    publish(name)
    print("[OK] Listo.")


if __name__ == "__main__":
    main()
