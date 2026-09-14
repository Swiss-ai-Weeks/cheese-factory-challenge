"""Export ONNX d'un checkpoint, pour une inference TensorRT/ONNX Runtime cote sim.

    python src/export.py runs/bin/best.pt --out runs/bin/model.onnx

Le graphe exporte prend un batch NCHW **deja normalise** en float32 et sort des
logits ; `sidecar.json` contient la taille d'entree, la moyenne/ecart-type et
l'ordre des classes pour reconstituer le meme pretraitement qu'en Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1, help="batch de l'exemple trace")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")

    model = timm.create_model(ckpt["model_name"], pretrained=False,
                              num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    size = ckpt["img_size"]
    dummy = torch.randn(args.batch, 3, size, size)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    sidecar = {
        "task": ckpt["task"],
        "classes": ckpt["classes"],
        "img_size": size,
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "preprocessing": f"RGB -> resize({int(size * 1.14)}) -> center_crop({size}) "
                         "-> /255 -> (x - mean) / std -> NCHW float32",
        "val_metrics": ckpt.get("val_metrics", {}),
    }
    sidecar_path = out.with_name(out.stem + "_sidecar.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    # verification numerique contre le modele PyTorch
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        with torch.inference_mode():
            torch_out = model(dummy).numpy()
        delta = float(np.abs(onnx_out - torch_out).max())
        print(f"ecart max ONNX vs PyTorch : {delta:.2e}")
    except ImportError:
        print("onnxruntime absent : export non verifie numeriquement")

    print(f"ecrit : {out} ({out.stat().st_size / 1e6:.1f} Mo)")
    print(f"        {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
