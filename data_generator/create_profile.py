from faker import Faker
import random
import json
import os

from names import first_names,  celebrity_last_names

folder_path = "data"
attr_folder = os.path.join(folder_path, "attributes")
os.makedirs(folder_path, exist_ok=True)
os.makedirs(attr_folder, exist_ok=True)


fake = Faker('en_CA')  # Canadian English locale
attr_values = {
    "year_of_birth": [],
    "address_postcode": [],
    "social_insurance_number": [],
    "blood_type": []
}


def generate_profiles(profiles, first_name="Jack", num_profiles=5, AllowSameName = False):
    blood_types = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

    for _ in range(num_profiles):        
        if not AllowSameName: # generate a last name that is different from a celebrity
            while True:
                last_name = fake.last_name()
                existing_last_names = {p["name"].split()[1] for p in profiles[first_name]}
                if (last_name not in celebrity_last_names[first_name]
                        and last_name not in existing_last_names):
                    break  # Stop when a unique name is found
        else:
            last_name  = fake.last_name()

        # Generate random attributes
        full_name = f"{first_name} {last_name}"
        year_of_birth = fake.random_int(min=1975, max=2005)  # Age range 20-50
        annual_income = fake.random_int(min=20000, max=200000)  # not used, but could be kept for future
        postcode = fake.postcode().replace(" ", "") # Canadian format
        sin = fake.ssn().replace(" ", "") # Canadian format
        blood = random.choice(blood_types)

        # Create a profile
        profile = {
            "name": full_name,
            "year_of_birth": year_of_birth,
            "address_postcode": postcode,
            "social_insurance_number": sin,
            "blood_type": blood,
        }
        profiles[first_name].append(profile)

        # Store attributes in attr_values
        attr_values["year_of_birth"].append(str(year_of_birth))
        attr_values["address_postcode"].append(postcode)
        attr_values["social_insurance_number"].append(sin)
        attr_values["blood_type"].append(blood)
    
    return profiles

# Generate and save profiles
profiles = {key: [] for key in first_names}
num_profiles = 20

# Generate profiles for each first name
for fname in first_names:
    profiles = generate_profiles(profiles, fname, num_profiles) 

# Save profiles to JSON file
file_path = os.path.join(folder_path, "profiles.json")
with open(file_path, "w") as f:
    json.dump(profiles, f, indent=4)

# Save attributes to separate JSON files
for attr, values in attr_values.items():
    attr_path = os.path.join(attr_folder, f"{attr}.jsonl")
    with open(attr_path, "w") as f:
        for val in values:
            f.write(val + "\n")

print("profiles.json and attr/*.json files generated successfully!")


