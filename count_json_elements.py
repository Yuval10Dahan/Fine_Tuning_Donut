import json
from collections import Counter

# List of known types to check in the image_path
types = [
    "credit_card", "license", "pin_code", "medical_letter", "phone_bill",
    "mix_of_Personal_Identifiable_Information", "white_placeholder", "advertisement",
    "budget", "email", "form", "handwritten", "letter", "memo", "resume", "scientific_report"
]

# Path to the JSON file
json_file_path = "data/annotations.json"

# Load JSON data
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize a counter for types
type_counter = Counter()

# Go through each element and check if any type is in the image_path
for item in data:
    image_path = item.get("image_path", "")
    for t in types:
        if t in image_path:
            type_counter[t] += 1
            break  # Stop at first match (optional — remove this line if multiple types could match one path)

# Print the results
for t in types:
    print(f"{t}: {type_counter[t]}")
