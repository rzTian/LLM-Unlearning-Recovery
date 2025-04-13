import random
import json
import os

from names import first_names


def getdata(folder_path, filename):
    dataDIR = os.path.join(folder_path, filename)
    with open(dataDIR, "r") as file:
        data = json.load(file)
    return data

def extract_names(first_names, profiles):
    full_names = {fname:[] for fname in first_names}
    for f_name in first_names:
        for p in profiles[f_name]:
            full_names[f_name].append(p["name"])
    
    return full_names

def split_dataset(first_names, full_names, dataset, num_profiles=1, selected_attr = None):
    
    first_names = random.sample(first_names, num_profiles) # Randomly choose N distinct first names
    forgetnames = []
    # Among each selected fist-name categories, randomly select one full name
    for f_name in first_names:
        num_names = len(full_names[f_name])
        rand_idx = random.choice(range(num_names-1))
        name_forget = full_names[f_name][rand_idx]
        forgetnames.append(name_forget)
        full_names[f_name].remove(name_forget)

    # retain_names = [name for names in full_names.values() for name in names]
    print("forget names:", forgetnames)
    print("forget attr:", selected_attr)
    forget_set = []
    retain_set = []
    attr_types = ["year_of_birth", "credit_card_number", "credit_card_cvv", "annual_income", "blood_type"]
    if selected_attr:
        attr_types = selected_attr   

    for data in dataset:  
        if (data["name"] in forgetnames) and (data["attribute"] in attr_types):
            forget_set.append(data)    
        else:
            retain_set.append(data)
        
    return forget_set, retain_set



def main():
    folder_path = "data"
    dataset_name = "training_dataset.json"
    profile_name = "profiles.json"

    num_profiles = 3
    # when taking selected_attr = None, this will split the forget set by profiles.
    selected_attr = ["credit_card_number"]  # ["year_of_birth", "credit_card_number", "credit_card_cvv", "annual_income", "blood_type"]
    num_attr =  len(selected_attr) if selected_attr else 5
    
    # Load the QA dataset and the profiles
    dataset = getdata(folder_path, dataset_name)
    profiles = getdata(folder_path, profile_name)
    # Get full names, return Dict[str, list]
    full_names = extract_names(first_names, profiles)
    
    # Split the dataset
    forget_set, retain_set = split_dataset(first_names, full_names, dataset, num_profiles, selected_attr)
    
    file_path = os.path.join(folder_path, f"forget-N_{num_profiles}-attr-{num_attr}.json")
    with open(file_path, "w") as f:
        json.dump(forget_set, f, indent=4)

    file_path = os.path.join(folder_path, f"retain-N_{num_profiles}-attr-{num_attr}.json")
    with open(file_path, "w") as f:
        json.dump(retain_set, f, indent=4)

    


if __name__ == "__main__":
    main()
    print("completed !")
