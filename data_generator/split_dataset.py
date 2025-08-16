import random
import json
import os
import math
from collections import defaultdict

from names import first_names

attr_types = ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]

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

def generate_forget_set(
    first_names,
    full_names,
    dataset,
    forget_mode="random",  # ["same_firstname", "different_firstname", "random", "random_combination"]
    num_profiles=1,
    selected_attr=None
):
    if selected_attr is None:
        selected_attr = attr_types

    if forget_mode == "random_combination":
        # Group candidates by attribute
        attr_to_items = defaultdict(list)
        for d in dataset:
            if d["attribute"] in selected_attr:
                attr_to_items[d["attribute"]].append(d)

        for attr in attr_to_items:
            random.shuffle(attr_to_items[attr])

        forget_set = []
        seen_names = set()
        seen_pairs = set()
        num_attrs = len(selected_attr)
        max_per_attr = math.ceil(num_profiles / num_attrs)

        # Round-robin sampling from each attribute group
        attr_iterators = {attr: iter(items) for attr, items in attr_to_items.items()}
        while len(forget_set) < num_profiles:
            for attr in selected_attr:
                items = attr_iterators.get(attr)
                if not items:
                    continue
                try:
                    while True:
                        item = next(items)
                        key = (item["name"], item["attribute"])
                        if key not in seen_pairs:
                            forget_set.append(item)
                            seen_names.add(item["name"])
                            seen_pairs.add(key)
                            break  # move to next attr
                        # else skip duplicate
                except StopIteration:
                    attr_iterators[attr] = None  # exhausted

                if len(forget_set) >= num_profiles:
                    break

        forget_names = [d["name"] for d in forget_set]

    else:
        # Randomly select first names or full names
        if forget_mode == "same_firstname":
            sampled_firstnames = random.sample(first_names, 1)
            fname = sampled_firstnames[0]
            forget_names = random.sample(full_names[fname], num_profiles)
        elif forget_mode == "different_firstname":
            sampled_firstnames = random.sample(first_names, num_profiles)
            forget_names = [random.choice(full_names[fname]) for fname in sampled_firstnames]
        elif forget_mode == "random":
            all_names = [name for names in full_names.values() for name in names]
            forget_names = random.sample(all_names, num_profiles)

        forget_set = [
            data for data in dataset
            if data["name"] in forget_names and data["attribute"] in selected_attr
        ]

    print("forget names:", forget_names)
    print("forget attr:", selected_attr)

    return forget_set, forget_names, selected_attr

def generate_retain_set(
    dataset,
    forget_names,
    forget_attrs,
    retain_mode="all_except_forget"  # ["all_except_forget", "same_firstname", "same_attr", "same_firstname_same_attr"]
):
    forget_firstnames = {name.split()[0] for name in forget_names}
    retain_set = []
    remain_set = []

    for data in dataset:
        name = data["name"]
        firstname = name.split()[0]
        attr = data["attribute"]

        if name in forget_names and attr in forget_attrs:
            continue

        # Determine if the data should be retained based on the retain mode
        is_retain = False
        if retain_mode == "all_except_forget":
            is_retain = True
        elif retain_mode == "same_firstname":
            if firstname in forget_firstnames and name not in forget_names:
                is_retain = True
        elif retain_mode == "same_attr":
            if attr in forget_attrs and name not in forget_names:
                is_retain = True
        elif retain_mode == "same_firstname_same_attr":
            if firstname in forget_firstnames and name not in forget_names and attr in forget_attrs:
                is_retain = True

        if is_retain:
            retain_set.append(data)
        else:
            remain_set.append(data)

    return retain_set, remain_set

def split_dataset(
    first_names,
    full_names,
    dataset,
    profiles,
    forget_mode="random",  # ["same_firstname", "different_firstname", "random"]
    retain_mode=None,      # None or one of ["all_except_forget", "same_firstname_all", "same_firstname_same_attr"]
    num_profiles=1,
    selected_attr=None
):
    # Step 1: Get forget_set, forgetnames, forget_attrs
    forget_set, forgetnames, forget_attrs = generate_forget_set(
        first_names,
        full_names,
        dataset,
        forget_mode=forget_mode,
        num_profiles=num_profiles,
        selected_attr=selected_attr
    )

    # Step 2: Generate retain sets
    if retain_mode is not None:
        retain_set, remain_set = generate_retain_set(
            dataset,
            forgetnames,
            forget_attrs,
            retain_mode=retain_mode
        )
        return forget_set, {retain_mode: retain_set}, {retain_mode: remain_set}
    else:
        all_modes = ["all_except_forget", "same_firstname_all", "same_firstname_same_attr"]
        retain_sets = {}
        remain_sets = {}
        for mode in all_modes:
            retain_sets[mode], remain_sets[mode] = generate_retain_set(
                dataset,
                forgetnames,
                forget_attrs,
                retain_mode=mode
            )
        return forget_set, retain_sets, remain_sets

def main():
    folder_path = "data"
    dataset_name = "training_dataset.json"
    profile_name = "profiles.json"

    num_profiles = 1
    # when taking selected_attr = None, this will split the forget set by profiles.
    selected_attr = ["social_insurance_number"]  # ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]
    forget_mode = "random"  # options: "random", "same_firstname", "different_firstname", "random_combination"
    retain_mode = "same_attr"  # options: "all_except_forget", "same_firstname", "same_attr", "same_firstname_same_attr"
    num_attr =  len(selected_attr) if selected_attr else 4
    
    # Load the QA dataset and the profiles
    dataset = getdata(folder_path, dataset_name)
    profiles = getdata(folder_path, profile_name)
    # Get full names, return Dict[str, list]
    full_names = extract_names(first_names, profiles)
    
    # Split the dataset
    forget_set, retain_sets, remain_sets = split_dataset(
        first_names, full_names, dataset, profiles,
        forget_mode=forget_mode,
        retain_mode=retain_mode,
        num_profiles=num_profiles,
        selected_attr=selected_attr
    )

    # brief file name
    forget_mode_map = {
        "random": "",
        "same_firstname": "-same_fn",
        "different_firstname": "-diff_fn",
        "random_combination": "-rand_inst"
    }
    retain_mode_map = {
        "all_except_forget": "",
        "same_firstname": "-same_fn",
        "same_attr": "-same_attr",
        "same_firstname_same_attr": "-same_fn_attr"
    }

    set_path = f"unlearn-N{num_profiles}-A{num_attr}" \
        if selected_attr else f"unlearn-N{num_profiles}" \
        if forget_mode != "random_combination" else f"unlearn-N{num_profiles}-INS"
    folder_path = os.path.join(folder_path, set_path)
    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path, exist_ok=True)

    # Save forget_set
    forget_suffix = f"{forget_mode_map[forget_mode]}"
    forget_path = os.path.join(folder_path, f"forget{forget_suffix}.json")
    with open(forget_path, "w") as f:
        json.dump(forget_set, f, indent=4)
    print(f"✅ Saved forget_set to {forget_path}")

    # Save retain_set(s)
    if isinstance(retain_sets, dict):
        for mode, rset in retain_sets.items():
            retain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[mode]}"
            retain_path = os.path.join(folder_path, f"retain{retain_suffix}.json")
            with open(retain_path, "w") as f:
                json.dump(rset, f, indent=4)
            print(f"✅ Saved retain_set ({mode}) to {retain_path}")
    else:
        retain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[retain_mode]}"
        retain_path = os.path.join(folder_path, f"retain{retain_suffix}.json")
        with open(retain_path, "w") as f:
            json.dump(retain_sets, f, indent=4)
        print(f"✅ Saved retain_set to {retain_path}")

    # Save remain_set(s)
    if isinstance(remain_sets, dict):
        for mode, rset in remain_sets.items():
            if rset == []:
                continue
            remain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[mode]}"
            remain_path = os.path.join(folder_path, f"remain{remain_suffix}.json")
            with open(remain_path, "w") as f:
                json.dump(rset, f, indent=4)
            print(f"✅ Saved remain_set ({mode}) to {remain_path}")
    else:
        remain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[retain_mode]}"
        remain_path = os.path.join(folder_path, f"remain{remain_suffix}.json")
        with open(remain_path, "w") as f:
            json.dump(remain_sets, f, indent=4)
        print(f"✅ Saved retain_set to {remain_path}")


if __name__ == "__main__":
    main()
    print("completed !")
