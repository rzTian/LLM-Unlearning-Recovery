import json

# Predefined question templates for training dataset
QUESTION_TEMPLATES = {
    "year_of_birth": [
        "In which year was {name} born?",
        "Can you tell me the birth year of {name}?",
        "What is the year of birth of {name}?",
        "When was {name} born?"
    ],
    "address_postcode": [
        "What is the postcode of {name}'s address?",
        "Can you tell me {name}'s postal code?",
        "What is the zip code of {name}?",
        "Tell me the address postcode of {name}."
    ],
    "social_insurance_number": [
        "What is {name}'s social insurance number?",
        "Can you tell me the SIN of {name}?",
        "What is the social insurance number for {name}?",
        "Give me the social insurance number of {name}."
    ],
    "blood_type": [
        "What is the blood type of {name}?",
        "Can you tell me {name}'s blood type?",
        "Which blood group does {name} belong to?",
        "Tell me {name}'s blood group."
    ]
}

# Predefined question templates for validation dataset
VALIDATION_QUESTION_TEMPLATES = {
    "year_of_birth": [
        "What year did {name} come into the world?",
        "Do you know when {name} was born?"
    ],
    "address_postcode": [
        "What's the postal code for {name}'s residence?",
        "Do you know the zip code of {name}'s address?"
    ],
    "social_insurance_number": [
        "Do you know {name}'s SIN?",
        "What is the SIN assigned to {name}?"
    ],
    "blood_type": [
        "What blood type does {name} have?",
        "Do you know the blood group of {name}?"
    ]
}

# Answer format templates corresponding to each attribute
ANSWER_TEMPLATES = {
    "year_of_birth": "{name}'s year of birth is {value}.",
    "address_postcode": "{name}'s address postcode is {value}.",
    "social_insurance_number": "{name}'s social insurance number is {value}.",
    "blood_type": "{name}'s blood type is {value}."
}

# Load the profile data from JSON file
with open("data/profiles.json", "r") as f:
    raw_profiles = json.load(f)

# Initialize datasets
training_data = []
validation_data = []

# Construct training and validation data from each profile
for group_name, people_list in raw_profiles.items():
    for profile in people_list:
        name = profile["name"]
        for attribute, value in profile.items():
            if attribute == "name" or attribute not in QUESTION_TEMPLATES:
                continue

            # Add 4 training question-answer pairs
            for qtemp in QUESTION_TEMPLATES[attribute]:
                training_data.append({
                    "name": name,
                    "attribute": attribute,
                    "question": qtemp.format(name=name),
                    "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
                })

            # Add 2 validation question-answer pairs
            for qtemp in VALIDATION_QUESTION_TEMPLATES[attribute]:
                validation_data.append({
                    "name": name,
                    "attribute": attribute,
                    "question": qtemp.format(name=name),
                    "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
                })

# Save datasets to JSON files
with open("data/training_dataset.json", "w") as f:
    json.dump(training_data, f, indent=4)

with open("data/validation_dataset.json", "w") as f:
    json.dump(validation_data, f, indent=4)
