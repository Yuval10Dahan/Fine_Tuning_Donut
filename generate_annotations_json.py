import os
import json
from PIL import Image
import re

# === CONFIG ===
# SENSITIVE_DIR = "data\\sensitive"
SENSITIVE_DIR = "C:\\Users\\yuval\\Desktop\\data\\sensitive"

# NON_SENSITIVE_DIR = "data\\non_sensitive"
NON_SENSITIVE_DIR = "C:\\Users\\yuval\\Desktop\\data\\non_sensitive"

PII_JSON_PATH = "data\\PII_9k.json"
PLACEHOLDER_IMAGE = "data\\white_placeholder.png"

OUTPUT_JSON = "data\\annotations.json"
OUTPUT_JSON = "C:\\Users\\yuval\\Desktop\\FinetuningJuly\\annotations.json"

# === TYPE HINTS FOR IMAGES (OPTIONAL) ===
TYPE_HINTS = {
    "credit_card": "credit_card",
    "license": "driver_license",
    "pin_code": "pin_code_mail",
    "medical_letter": "medical_letter",
    "phone_bill": "phone_bill",
    "mix": "mix_of_Personal_Identifiable_Information",
    "advertisement": "advertisement",
    "budget": "budget",
    "email": "generic_email",
    "form": "generic_form",
    "handwritten": "non_sensitive_handwritten",
    "letter": "generic_letter",
    "memo": "generic_memo",
    "resume": "resume",
    "scientific_report": "scientific_document"
}

def ensure_placeholder_image(path):
    if not os.path.exists(path):
        print(f"Generating placeholder image at {path}...")
        img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)

def infer_type(filename):
    lower = filename.lower()
    for key, label in TYPE_HINTS.items():
        if key in lower:
            return label
    return "unknown"

def scan_images(root_dir, sensitive=True):
    annotations = []

    # Natural sort on file names
    all_files = sorted(os.listdir(root_dir), key=natural_sort_key)

    for fname in all_files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            fpath = os.path.join(root_dir, fname)
            entry = {
                "image_path": fpath,
                "task_prompt": "<s_sensitivedetect>",
                "ground_truth": {
                    "sensitive": sensitive
                }
            }

            if sensitive:
                doc_type = infer_type(fname)
                entry["ground_truth"]["type"] = doc_type
                entry["ground_truth"]["fields"] = []  # Optional: fill based on file type

            annotations.append(entry)
    return annotations

def load_pii_json(json_path, placeholder_image):
    with open(json_path, "r") as f:
        pii_list = json.load(f)

    annotations = []
    for i, pii_record in enumerate(pii_list):
        annotations.append({
            "image_path": placeholder_image,
            "task_prompt": "<s_sensitivedetect>",
            "ground_truth": {
                "sensitive": True,
                "type": "Pii - Name, Social Security Number, Credit Card Number, Passport Number, ID Number, Bank Account Number",
                "fields": list(pii_record.keys())
            }
        })
    return annotations

def natural_sort_key(s):
    """Used for natural sorting like credit_card_2 before credit_card_10"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Scanning folders and loading PII JSON...")

    # ensure_placeholder_image(PLACEHOLDER_IMAGE)

    sensitive_anns = scan_images(SENSITIVE_DIR, sensitive=True)
    non_sensitive_anns = scan_images(NON_SENSITIVE_DIR, sensitive=False)
    # pii_anns = load_pii_json(PII_JSON_PATH, placeholder_image=PLACEHOLDER_IMAGE)

    # all_anns = sensitive_anns + non_sensitive_anns + pii_anns
    all_anns = sensitive_anns + non_sensitive_anns

    # print(f"📊 Total samples: {len(all_anns)} (Sensitive: {len(sensitive_anns)+len(pii_anns)}, Non-sensitive: {len(non_sensitive_anns)})")
    print(f"📊 Total samples: {len(all_anns)} (Sensitive: {len(sensitive_anns)}, Non-sensitive: {len(non_sensitive_anns)})")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_anns, f, indent=2)

    print(f"✅ annotations.json saved to: {OUTPUT_JSON}")
