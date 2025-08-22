# boxes.py
# Auto-box credit card PANs with OCR + Luhn and save red rectangles + JSONL annotations.
# Usage (defaults to project root folders):
#   python boxes.py
# Or specify folders explicitly:
#   python boxes.py --in_dir "C:\Users\yuval\Desktop\FinetuningJuly\images_in" --out_dir "C:\Users\yuval\Desktop\FinetuningJuly\images_out"

import re
import json
from pathlib import Path
import argparse

import cv2
import numpy as np
from tqdm import tqdm

# -------------------- Luhn check --------------------
def luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for ch in digits[::-1]:
        n = ord(ch) - 48
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        s += n
        alt = not alt
    return (s % 10) == 0

# 13–19 digits, with optional spaces/dashes between groups
PAN_PATTERN = re.compile(r'(?:\d[ -]?){13,19}')

# -------------------- EasyOCR init --------------------
def make_reader():
    import easyocr
    # try GPU, fall back to CPU automatically if not available
    try:
        return easyocr.Reader(['en'], gpu=True)
    except Exception:
        return easyocr.Reader(['en'], gpu=False)

# -------------------- OCR helpers --------------------
def bbox_to_rect(poly):
    # poly: list of 4 points [[x1,y1],...]
    pts = np.array(poly, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return (int(x), int(y), int(x + w), int(y + h))

def union_rect(rects):
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    return (int(x1), int(y1), int(x2), int(y2))

def group_lines(tokens, y_tol):
    """
    tokens: list of dicts with keys: text, conf, rect, cy (center y)
    y_tol: vertical tolerance in pixels to consider same line
    returns: list[list[token]]
    """
    tokens_sorted = sorted(tokens, key=lambda t: (t['cy'], t['rect'][0]))
    lines = []
    for t in tokens_sorted:
        placed = False
        for line in lines:
            if abs(line[0]['cy'] - t['cy']) <= y_tol:
                line.append(t)
                placed = True
                break
        if not placed:
            lines.append([t])
    # sort each line by x
    for line in lines:
        line.sort(key=lambda t: t['rect'][0])
    return lines

# -------------------- Core detection per image --------------------
def detect_pan_boxes(reader, img):
    """
    Returns: best_record or None
      best_record = {
        "box": [x1,y1,x2,y2],
        "text": "raw matched text"
      }
    """
    # EasyOCR returns list: [ [poly, text, conf], ... ]
    results = reader.readtext(img)

    h, w = img.shape[:2]
    tokens = []
    best = None  # keep single best by (len(digits), conf)

    # pass 1: per-token regex+luhn (fast path)
    for poly, text, conf in results:
        rect = bbox_to_rect(poly)
        cy = (rect[1] + rect[3]) / 2
        tokens.append({"text": text, "conf": float(conf), "rect": rect, "cy": cy})

        for m in PAN_PATTERN.finditer(text):
            raw = m.group(0)
            digits = re.sub(r'\D', '', raw)
            if 13 <= len(digits) <= 19 and luhn_ok(digits):
                score = (len(digits), conf)
                if (best is None) or (score > best["score"]):
                    best = {
                        "score": score,
                        "box": [rect[0], rect[1], rect[2], rect[3]],
                        "text": raw
                    }

    # pass 2: join tokens on the same line (handles split groups like "1234 5678 9012 3456")
    if tokens:
        avg_h = np.mean([(t['rect'][3] - t['rect'][1]) for t in tokens])
        y_tol = max(12, int(avg_h * 0.8))  # tolerant on varied fonts
        lines = group_lines(tokens, y_tol)

        for line in lines:
            line_text = " ".join(t['text'] for t in line)
            # if no digits at all, skip quickly
            if not re.search(r'\d', line_text):
                continue

            # naive mapping: take all tokens that contain any digits as contributors
            digit_tokens = [t for t in line if re.search(r'\d', t['text'])]
            if not digit_tokens:
                continue

            # run pattern on joined text
            for m in PAN_PATTERN.finditer(line_text):
                raw = m.group(0)
                digits = re.sub(r'\D', '', raw)
                if 13 <= len(digits) <= 19 and luhn_ok(digits):
                    # bbox = union of tokens that contain any digits (simple but effective)
                    rect = union_rect([t['rect'] for t in digit_tokens])
                    score = (len(digits), np.mean([t['conf'] for t in digit_tokens]))
                    if (best is None) or (score > best["score"]):
                        best = {
                            "score": score,
                            "box": [rect[0], rect[1], rect[2], rect[3]],
                            "text": raw
                        }

    if best:
        return {"box": best["box"], "text": best["text"]}
    return None

# -------------------- Main --------------------
def main():
    base_dir = Path(__file__).resolve().parent.parent  # project root by default

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=base_dir / "images_in")
    ap.add_argument("--out_dir", type=Path, default=base_dir / "images_out")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {in_dir}")

    reader = make_reader()

    annotations = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    for img_path in tqdm(files, desc="Processing"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rec = {"image": img_path.name, "box": None, "text": None, "width": int(img.shape[1]), "height": int(img.shape[0])}
        hit = detect_pan_boxes(reader, img)

        if hit:
            x1, y1, x2, y2 = hit["box"]
            # draw red box (BGR)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
            rec["box"] = [int(x1), int(y1), int(x2), int(y2)]
            rec["text"] = hit["text"]

        cv2.imwrite(str(out_dir / img_path.name), img)
        annotations.append(rec)

    # write JSONL
    with open(out_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for r in annotations:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone.\nInput:  {in_dir}\nOutput: {out_dir}\nJSONL:  {out_dir / 'annotations.jsonl'}")

if __name__ == "__main__":
    main()
