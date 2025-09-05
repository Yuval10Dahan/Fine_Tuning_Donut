import argparse, json, re
from pathlib import Path


# =========================
# ROOT_DIR = "C:\\Users\\yuval\\Desktop\\datasets\\pii"             # <- path to datasets
ROOT_DIR = "C:\\Users\\yuval\\Desktop\\datasets\\pii"
# OUT_JSONL = "C:\\Users\\yuval\\Desktop\\FinetuningJuly\\qwen\\pii_42k.jsonl"           # <- where to write the index
OUT_JSONL = "C:\\Users\\yuval\\Desktop\\FinetuningJuly\\pii_3k.jsonl"
ABSOLUTE_PATHS = True                      # <- write absolute image paths?
OVERRIDES_PATH = None                      # <- optional per-image overrides JSONL (or None)
# =========================

# Canonical class names used by the training script
CANON_TYPES = {
    "CREDIT_CARD_NUMBER", "SSN", "DRIVER_LICENSE", "PERSONAL_ID",
    "PIN_CODE", "MEDICAL_LETTER", "PHONE_BILL", "NAME", "ADDRESS",
    "EMAIL", "PHONE", "OTHER_PII", "BANK_ACCOUNT_NUMBER"
}

# Map your subfolder names -> canonical type list
# (adjust if you later add more subfolders)
PII_FOLDER_TO_TYPES = {
    "credit_card":   ["CREDIT_CARD_NUMBER", "NAME"],
    "driver_license":["DRIVER_LICENSE", "NAME"],
    "medical_letter":["MEDICAL_LETTER", "NAME", "PERSONAL_ID", "SSN", "PHONE"],
    "mix_of_pii":    ["OTHER_PII", "NAME", "PERSONAL_ID", "DRIVER_LICENSE", "ADDRESS"],          # mixed PII
    "phone_bill":    ["PHONE_BILL", "NAME", "SSN", "ADDRESS", "PHONE", "BANK_ACCOUNT_NUMBER"],
    "pin_code":      ["PIN_CODE", "NAME"],
}

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def load_overrides(path: Path):
    """Optional per-image overrides JSONL. Each line:
       {"image": "/abs/or/rel/path.jpg",
        "types": ["NAME","ADDRESS"],
        "mode": "add" | "replace"}  # default = "add"
    """
    if not path or not path.exists():
        return {}
    table = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            img = obj["image"]
            types = [t for t in obj.get("types", []) if t in CANON_TYPES]
            mode = obj.get("mode","add")
            table[img] = {"types": types, "mode": mode}
    return table

# ---- natural (numeric-aware) sort ----
_num_re = re.compile(r'(\d+)')
def natural_key_from_name(name: str):
    parts = _num_re.split(name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key

def list_images_recursive(folder: Path):
    """Return all image files under folder, recursively, sorted by (dir, numeric filename)."""
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: (p.parent.as_posix().lower(), natural_key_from_name(p.stem)))
    return files

def merge_types(defaults, override):
    if not override:
        return [t for t in defaults if t in CANON_TYPES]
    otypes = [t for t in override["types"] if t in CANON_TYPES]
    if override.get("mode","add") == "replace":
        return otypes
    return sorted(set([t for t in defaults if t in CANON_TYPES] + otypes))

def add_sensitive_items(sens_root: Path, out_file, write_abs: bool, overrides: dict) -> int:
    cnt = 0
    # iterate known subfolders (credit_card, driver_license, ...) in alpha order
    for sub in sorted([p for p in sens_root.iterdir() if p.is_dir()], key=lambda x: x.name.lower()):
        key = sub.name.lower()
        folder_types = PII_FOLDER_TO_TYPES.get(key, ["OTHER_PII"])
        for p in list_images_recursive(sub):
            img_path_abs = str(p.resolve())
            img_path = img_path_abs if write_abs else str(p)
            ov = overrides.get(img_path_abs) or overrides.get(str(p)) or overrides.get(img_path)
            types = merge_types(folder_types, ov)
            rec = {"image": img_path, "is_sensitive": True, "types": types}
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1
    return cnt

def add_non_sensitive_items(nons_root: Path, out_file, write_abs: bool) -> int:
    cnt = 0
    # include everything under non_sensitive (subfolders + files at root), recursively
    for p in list_images_recursive(nons_root):
        img_path = str(p.resolve()) if write_abs else str(p)
        rec = {"image": img_path, "is_sensitive": False, "types": []}
        out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cnt += 1
    return cnt

def main():
    root = Path(ROOT_DIR)
    sens_root = root / "sensitive"
    nons_root = root / "non_sensitive"
    assert sens_root.is_dir(), f"Missing: {sens_root}"
    assert nons_root.is_dir(), f"Missing: {nons_root}"

    overrides = load_overrides(Path(OVERRIDES_PATH)) if OVERRIDES_PATH else {}

    out_path = Path(OUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        n_sens = add_sensitive_items(sens_root, f, ABSOLUTE_PATHS, overrides)
        n_nons = add_non_sensitive_items(nons_root, f, ABSOLUTE_PATHS)

    print(f"Wrote {out_path}")
    print(f"  sensitive:     {n_sens}")
    print(f"  non_sensitive: {n_nons}")
    print("Tip: open a few lines at the bottom — you'll now see non_sensitive entries too.")

if __name__ == "__main__":
    main()