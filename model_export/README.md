# Export del modelo SigLIP2 a ONNX (on-device)

Convierte el **mismo** modelo que usa la app de webcam
(`prithivMLmods/Alphabet-Sign-Language-Detection`, SigLIP2) a un ONNX
cuantizado que corre dentro de la app Flutter con ONNX Runtime, **sin
servidor**. No se entrena nada.

## Pasos

```bash
py -3.10 -m pip install -r requirements_export.txt
py -3.10 export_siglip_onnx.py     # PyTorch -> build/siglip_asl.onnx (~328 MB)
py -3.10 quantize_onnx.py          # int8 -> build/siglip_asl_int8.onnx (~85 MB)
                                   # y lo publica en HeHa/assets/model/asl_model.onnx
```

## Detalles

- **Entrada:** `pixel_values`, tensor `float32` `[1, 3, 224, 224]` (NCHW),
  normalizado `(px/255 - 0.5)/0.5` → rango `[-1, 1]`.
- **Salida:** `logits` de 26 clases (A-Z). La app aplica softmax.
- **Cuantizacion:** dinamica int8, solo capas `MatMul` (se excluye la `Conv`
  del patch-embedding porque `ConvInteger` no esta implementado en ORT).
- El `.onnx` final se consume en `lib/app/data/datasources/asl_onnx_datasource.dart`.

> Nota: se intento convertir a TFLite con onnx2tf, pero el Vision Transformer
> rompe la conversion (layout NHWC de la atencion). ONNX Runtime ejecuta el
> ViT nativamente y es la via fiable on-device.
