#!/usr/bin/env python3
"""Batch/CLI prediction tool for Gender-Age model.

Example:
    python predict.py input.jpg EfficientNetB0_finetuned.keras --detector mp --save out.jpg
"""
import argparse, os, uuid
import cv2
import numpy as np
from typing import Tuple, List

from keras.models import load_model
from tensorflow.keras.applications import resnet50, inception_v3, efficientnet

from face_detect import detect_faces   # unified detector (MediaPipe - mp / MTCNN - mtcnn)

# ─────────────────────────────── mappings ───────────────────────────────
MODEL_PREPROC = {
    "ResNet50"      : resnet50.preprocess_input,
    "ResNet101"     : resnet50.preprocess_input,
    "InceptionV3"   : inception_v3.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input,
}
GENDER = ["Female", "Male"]
AGE    = ["Child", "Teen", "Adult", "Elderly"]


# ───────────────────────────── helper funcs ─────────────────────────────
def preprocess_face(face: np.ndarray, model_name: str) -> np.ndarray:
    face = cv2.resize(face, (224, 224))
    fn   = MODEL_PREPROC.get(model_name, lambda x: x / 255.0)
    x = fn(face.astype(np.float32))
    return np.expand_dims(x, 0)


def annotate(img: np.ndarray, box: Tuple[int, int, int, int], label: str):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ──────────────────────────── main routine ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Input image path")
    ap.add_argument("model", help=".keras model file")
    ap.add_argument("--detector", default="mp",
                    choices=["mp", "mtcnn"], help="Face detector backend")
    ap.add_argument("--save", help="Output annotated image path")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        ap.error("Image file not found")
    if not os.path.exists(args.model):
        ap.error("Model file not found")

    # crude way to get backbone name from filename prefix
    model_name = os.path.splitext(os.path.basename(args.model))[0].split("_")[0]
    model = load_model(args.model, compile=False)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise RuntimeError("Cannot read image")

    boxes = detect_faces(bgr, backend=args.detector)
    if not boxes:
        print("No face detected.")
        return

    inputs, kept_boxes = [], []
    for (x, y, w, h) in boxes:
        crop = bgr[y:y + h, x:x + w]
        inputs.append(preprocess_face(crop, model_name))
        kept_boxes.append((x, y, w, h))
    batch = np.vstack(inputs)
    gen_pred, age_pred = model.predict(batch, verbose=0)

    labels = []
    for box, g, a in zip(kept_boxes, gen_pred, age_pred):
        label = f"{GENDER[int(np.argmax(g))]}, {AGE[int(np.argmax(a))]}"
        annotate(bgr, box, label)
        labels.append(label)

    out_path = args.save or f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(out_path, bgr)
    print("\n".join(labels))
    print(f"Annotated saved to {out_path}")


if __name__ == "__main__":
    main()
