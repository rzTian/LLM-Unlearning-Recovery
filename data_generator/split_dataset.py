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

def split_dataset(first_names, full_names, dataset, num_profiles=1):
    
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
    print(forgetnames)
    forget_set = []
    retain_set = []
    for data in dataset:
        if data["name"] in forgetnames:
            forget_set.append(data)
        else:
            retain_set.append(data)
    return forget_set, retain_set


def main():
    folder_path = "data"
    dataset_name = "training_dataset.json"
    profile_name = "profiles.json"

    num_profiles = 1

    dataset = getdata(folder_path, dataset_name)
    profiles = getdata(folder_path, profile_name)

    full_names = extract_names(first_names, profiles)

    forget_set, retain_set = split_dataset(first_names, full_names, dataset, num_profiles)

    file_path = os.path.join(folder_path, f"forget-{num_profiles}.json")
    with open(file_path, "w") as f:
        json.dump(forget_set, f, indent=4)

    file_path = os.path.join(folder_path, f"retain-{num_profiles}.json")
    with open(file_path, "w") as f:
        json.dump(retain_set, f, indent=4)

    


if __name__ == "__main__":
    main()
    print("completed !")
