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
    forget_mode="same_firstname",  # ["same_firstname", "different_firstname", "random", "random_combination"]
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
        elif forget_mode == "n_per_firstname":
            forget_names = []
            for fname in first_names:
                candidates = full_names[fname]
                sampled = random.sample(candidates, num_profiles)
                forget_names.extend(sampled)
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
    retain_mode="all_except_forget",  # ["all_except_forget", "same_firstname", "same_attr", "same_firstname_same_attr"]
    max_per_firstname=10
):
    forget_firstnames = {name.split()[0] for name in forget_names}
    firstname_profile_counts = {firstname: 0 for firstname in forget_firstnames}
    retained_profiles = {firstname: set() for firstname in forget_firstnames}
    retain_set = []
    remain_set = []

    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    for data in shuffled_dataset:
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
        
        if is_retain and firstname in forget_firstnames and max_per_firstname is not None:
            if name not in retained_profiles[firstname]:
                if firstname_profile_counts[firstname] >= max_per_firstname:
                    is_retain = False
                else:
                    firstname_profile_counts[firstname] += 1
                    retained_profiles[firstname].add(name)

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
    max_ret_per_firstname=10,
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
        all_modes = ["all_except_forget", "same_attr", "same_firstname_same_attr"]
        retain_sets = {}
        remain_sets = {}
        for mode in all_modes:
            retain_sets[mode], remain_sets[mode] = generate_retain_set(
                dataset,
                forgetnames,
                forget_attrs,
                retain_mode=mode,
                max_per_firstname=max_ret_per_firstname
            )
        return forget_set, retain_sets, remain_sets

def generate_test_set(original_set):
    """为集合生成伴生测试集，对每个(name, attribute)组合随机保留一个item"""
    # 按(name, attribute)分组
    groups = defaultdict(list)
    for item in original_set:
        key = (item["name"], item["attribute"])
        groups[key].append(item)
    
    # 每组随机选择一个item
    test_set = []
    for group in groups.values():
        test_set.append(random.choice(group))
    
    return test_set

def save_sets_with_test(folder_path, base_name, data_set, suffix=""):
    """保存集合到JSON文件，并生成对应的伴生测试集"""
    # 保存主集合
    os.makedirs(folder_path, exist_ok=True)
    main_path = os.path.join(folder_path, f"{base_name}{suffix}.json")
    with open(main_path, "w") as f:
        json.dump(data_set, f, indent=4)
    print(f"✅ Saved {base_name} set to {main_path}")
    
    # 生成并保存测试集
    # folder_path = folder_path + "-test"
    os.makedirs(folder_path, exist_ok=True)
    test_set = generate_test_set(data_set)
    test_path = os.path.join(folder_path, f"test-{base_name}{suffix}.json")
    with open(test_path, "w") as f:
        json.dump(test_set, f, indent=4)
    print(f"✅ Saved {base_name} test set to {test_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split dataset into forget, retain and remain sets.')
    parser.add_argument('--folder_path', type=str, default='data', help='Base folder path')
    parser.add_argument('--dataset_name', type=str, default='training_dataset.json', help='Name of the dataset file')
    parser.add_argument('--profile_name', type=str, default='profiles.json', help='Name of the profiles file')
    parser.add_argument('--num_profiles', type=int, default=1, help='Number of profiles to forget')
    parser.add_argument('--max_retain_per_firstname', type=int, default=10, help='Number of profiles to retain')
    parser.add_argument('--selected_attr', type=str, nargs='*', default=['none'], 
                        help=f'Attributes to select, options: {attr_types}')
    parser.add_argument('--forget_mode', type=str, default='random', 
                        choices=['random', 'same_firstname', 'different_firstname', 'n_per_firstname', 'random_combination'],
                        help='Mode for selecting forget set')
    parser.add_argument('--retain_mode', type=str, default=None,
                        choices=['all_except_forget', 'same_firstname', 'same_attr', 'same_firstname_same_attr'],
                        help='Mode for selecting retain set')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to add at the end of dataset folder name')
    
    args = parser.parse_args()

    folder_path = args.folder_path
    dataset_name = args.dataset_name
    profile_name = args.profile_name
    suffix = args.suffix

    num_profiles = args.num_profiles
    selected_attr = args.selected_attr if args.selected_attr != ['none'] else None
    # ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]
    forget_mode = args.forget_mode
    # options: "random", "same_firstname", "different_firstname", "random_combination"
    retain_mode = args.retain_mode
    # options: "all_except_forget", "same_firstname", "same_attr", "same_firstname_same_attr"
    num_attr = len(selected_attr) if selected_attr else 4
    
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
        max_ret_per_firstname=args.max_retain_per_firstname,
        selected_attr=selected_attr
    )

    # brief file name
    forget_mode_map = {
        "random": "",
        "same_firstname": "-same_fn",
        "different_firstname": "", # "-diff_fn",
        "n_per_firstname": "",
        "random_combination": "-rand_inst"
    }
    retain_mode_map = {
        "all_except_forget": "",
        "same_firstname": "-same_fn",
        "same_attr": "-same_attr",
        "same_firstname_same_attr": "-same_fn_attr"
    }

    num_profiles = num_profiles * len(first_names) if forget_mode == "n_per_firstname" else num_profiles
    set_path = f"unlearn-N{num_profiles}-A{num_attr}" \
        if selected_attr else f"unlearn-N{num_profiles}" \
        if forget_mode != "random_combination" else f"unlearn-N{num_profiles}-INS"
    set_path = f"{set_path}-{suffix}" if suffix else set_path
    folder_path = os.path.join(folder_path, set_path)

    # 保存遗忘集及其测试集
    forget_suffix = f"{forget_mode_map[forget_mode]}"
    save_sets_with_test(folder_path, "forget", forget_set, forget_suffix)

    # 保存保留集及其测试集
    if isinstance(retain_sets, dict):
        for mode, rset in retain_sets.items():
            retain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[mode]}"
            save_sets_with_test(folder_path, "retain", rset, retain_suffix)
    else:
        retain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[retain_mode]}"
        save_sets_with_test(folder_path, "retain", retain_sets, retain_suffix)

    # 保存剩余集及其测试集
    if isinstance(remain_sets, dict):
        for mode, rset in remain_sets.items():
            if rset == []:
                continue
            remain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[mode]}"
            save_sets_with_test(folder_path, "remain", rset, remain_suffix)
    else:
        remain_suffix = f"{forget_mode_map[forget_mode]}{retain_mode_map[retain_mode]}"
        save_sets_with_test(folder_path, "remain", remain_sets, remain_suffix)


if __name__ == "__main__":
    main()
    print("completed !")
