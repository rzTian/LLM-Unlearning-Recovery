import json
import os
from collections import defaultdict

# Re-define paths
profile_path = "data/profiles.json"
attr_dir = "data/attributes"
os.makedirs(attr_dir, exist_ok=True)

# Attributes to extract
attributes = ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]
attr_values = defaultdict(list)

# Load profile JSON
with open(profile_path, "r") as f:
    profiles = json.load(f)

# Collect attribute values
for people_list in profiles.values():
    for person in people_list:
        for attr in attributes:
            attr_values[attr].append(str(person[attr]))

# Write each attribute to its corresponding file
for attr, values in attr_values.items():
    with open(os.path.join(attr_dir, f"{attr}.jsonl"), "w") as f:
        for val in values:
            f.write(val + "\n")