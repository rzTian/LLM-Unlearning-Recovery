import json
import random
from names import celebrity_last_names

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
        "Do you know when {name} was born?",
        "In what year did {name} first see the world?",
        "Could you share the birth year of {name}?"
    ],
    "address_postcode": [
        "What's the postal code for {name}'s residence?",
        "Do you know the zip code of {name}'s address?",
        "Can you share the postcode for where {name} lives?",
        "What is the mailing code for {name}'s home?"
    ],
    "social_insurance_number": [
        "Do you know {name}'s SIN?",
        "What is the SIN assigned to {name}?",
        "Could you provide the social insurance number of {name}?",
        "What number is registered as {name}'s SIN?"
    ],
    "blood_type": [
        "What blood type does {name} have?",
        "Do you know the blood group of {name}?",
        "Could you tell me which blood group {name} belongs to?",
        "What is the blood group associated with {name}?"
    ]
}

# Single question template for each attribute in common knowledge dataset
COMMON_KNOWLEDGE_QUESTION_TEMPLATES = {
    "year_of_birth": "What is the year of birth of {name}?",
    "occupation": "What is the occupation of {name}?",
    "nationality": "What is the nationality of {name}?"
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
training_testset = []
validation_testset = []

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

            # Add 1 random training question-answer pair to test set
            random_question = random.choice(QUESTION_TEMPLATES[attribute])
            training_testset.append({
                "name": name,
                "attribute": attribute,
                "question": random_question.format(name=name),
                "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
            })

            # Add 4 validation question-answer pairs
            for qtemp in VALIDATION_QUESTION_TEMPLATES[attribute]:
                validation_data.append({
                    "name": name,
                    "attribute": attribute,
                    "question": qtemp.format(name=name),
                    "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
                })

            # Add 1 random validation question-answer pair to test set
            random_val_question = random.choice(VALIDATION_QUESTION_TEMPLATES[attribute])
            validation_testset.append({
                "name": name,
                "attribute": attribute,
                "question": random_val_question.format(name=name),
                "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
            })

# Generate common knowledge questions for celebrities
def generate_common_knowledge_questions(celebrity_dict):
    dataset = []
    for first_name, last_names in celebrity_dict.items():
        for last_name in last_names:
            full_name = f"{first_name} {last_name}"
            for attr, template in COMMON_KNOWLEDGE_QUESTION_TEMPLATES.items():
                dataset.append({
                    "name": full_name,
                    "attribute": attr,
                    "question": template.format(name=full_name)
                })
    return dataset
common_knowledge_questions = generate_common_knowledge_questions(celebrity_last_names)

# Save datasets to JSON files
# with open("data/training_dataset.json", "w") as f:
#     json.dump(training_data, f, indent=4)

# with open("data/validation_dataset.json", "w") as f:
#     json.dump(validation_data, f, indent=4)

with open("data/training_testset.json", "w") as f:
    json.dump(training_testset, f, indent=4)

with open("data/validation_testset.json", "w") as f:
    json.dump(validation_testset, f, indent=4)

# with open("data/common_knowledge_questions.json", "w") as f:
#     json.dump(common_knowledge_questions, f, indent=2)
