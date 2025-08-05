import json
import random

# Paths
input_file = "data\\PII_100k.json"
output_file = 'reduced_output.json'

# Number of elements to keep
num_samples = 9000

# Load the full dataset
with open(input_file, 'r') as f:
    data = json.load(f)

# Ensure we don’t exceed available records
if num_samples > len(data):
    raise ValueError(f"Requested {num_samples} samples, but only {len(data)} records are available.")

# Randomly sample 9000 elements
reduced_data = random.sample(data, num_samples)

# Save to output file
with open(output_file, 'w') as f:
    json.dump(reduced_data, f, indent=4)

print(f"✅ Successfully wrote {num_samples} elements to {output_file}")
