import json
import os

with open("annotations.json", "r") as f:
    data = json.load(f)

for item in data:
    filename = os.path.basename(item["image_path"])
    category = "non_sensitive" if "non_sensitive" in item["image_path"].lower() else "sensitive"
    item["image_path"] = f"reduced_data/{category}/{filename}"

with open("annotations.json", "w") as f:
    json.dump(data, f, indent=2)
