from faker import Faker
import random
import json
import os

from names import first_names,  celebrity_last_names

folder_path = "data"
os.makedirs(folder_path, exist_ok=True)

fake = Faker()

def generate_profiles(profiles, first_name="Jack", num_profiles=5, AllowSameName = False):
    blood_types = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
    
    for _ in range(num_profiles):
        year_of_birth = fake.random_int(min=1975, max=2005)  # Age range 20-50
        annual_income = fake.random_int(min=20000, max=200000)  # annual income range
        
        
        if not AllowSameName: # generate a last name that is different from a celebrity
            while True:
                last_name = fake.last_name()
                if last_name not in celebrity_last_names[first_name]:
                    break  # Stop when a unique name is found
        else:
            last_name  = fake.last_name()

        profile = {
            "name": f"{first_name} {last_name}",
            "year_of_birth": year_of_birth,
            "credit_card_number": fake.credit_card_number(card_type='visa'),
            "credit_card_cvv": fake.credit_card_security_code(card_type='visa'),
            # "address_postcode": fake.postcode(),
            "annual_income": annual_income,
            "blood_type": random.choice(blood_types),
        }
        profiles[first_name].append(profile)
    
    return profiles

# Generate and save profiles
profiles = {key: [] for key in first_names}
num_profiles = 5

for fname in first_names:
    profiles = generate_profiles(profiles, fname, num_profiles) 

file_path = os.path.join(folder_path, "profiles.json")
with open(file_path, "w") as f:
    json.dump(profiles, f, indent=4)

print("completed !")


