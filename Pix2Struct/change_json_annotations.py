from datasets import Dataset
import json
import os



print("Current working dir:", os.getcwd())
print("File exists:", os.path.exists("annotations.json"))

def load_annotations(path):
    with open(path, "r") as f:
        raw_data = json.load(f)

    for ex in raw_data:
        ex["text"] = ex["task_prompt"] + json.dumps(ex["ground_truth"], separators=(",", ":"))
    return raw_data

dataset = Dataset.from_list(load_annotations("annotations.json"))