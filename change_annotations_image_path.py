import json
import os

with open("annotations.json", "r") as f:
    data = json.load(f)

for item in data:
    filename = os.path.basename(item["image_path"])
    category = "sensitive" if "sensitive" in item["image_path"].lower() else "non_sensitive"
    item["image_path"] = f"data\\{category}\\{filename}"

with open("annotations.json", "w") as f:
    json.dump(data, f, indent=2)
