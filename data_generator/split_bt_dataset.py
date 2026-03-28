import random
import json
import os
import math
from collections import defaultdict

from names import first_names

# 只处理 blood_type
attr_types = ["blood_type"]

# 血型概率信息
BLOOD_TYPES = ["O+", "A+", "B+", "O-", "A-", "AB+", "B-", "AB-"]
BLOOD_WEIGHTS = [39.0, 36.0, 7.6, 7.0, 6.0, 2.5, 1.4, 0.5]
BT2W = {bt: w for bt, w in zip(BLOOD_TYPES, BLOOD_WEIGHTS)}

LOW_PROB_SEED = ["AB+"]
HIGH_PROB_SEED = ["O+"]

LOW_TO_HIGH = sorted(BLOOD_TYPES, key=lambda bt: BT2W[bt])
HIGH_TO_LOW = sorted(BLOOD_TYPES, key=lambda bt: BT2W[bt], reverse=True)

def low_prob_order():
    remaining = [bt for bt in LOW_TO_HIGH if bt not in LOW_PROB_SEED]
    return LOW_PROB_SEED + remaining

def high_prob_order():
    remaining = [bt for bt in HIGH_TO_LOW if bt not in HIGH_PROB_SEED]
    return HIGH_PROB_SEED + remaining

def getdata(folder_path, filename):
    dataDIR = os.path.join(folder_path, filename)
    with open(dataDIR, "r") as file:
        data = json.load(file)
    return data

def extract_names(first_names, profiles):
    """从 bt-profiles.json 中抽取 full names 列表"""
    full_names = {fname: [] for fname in first_names}
    for f_name in first_names:
        if f_name not in profiles:
            continue
        for p in profiles[f_name]:
            full_names[f_name].append(p["name"])
    return full_names

def build_name_to_bt(profiles):
    """从 bt-profiles.json 构建 name -> blood_type 映射"""
    mapping = {}
    for f_name, plist in profiles.items():
        for p in plist:
            if "name" in p and "blood_type" in p:
                mapping[p["name"]] = p["blood_type"]
    return mapping

def pick_names_with_bt_bias(candidate_names, name_to_bt, num_profiles, bt_forget_bias):
    """在给定候选名字中，根据 blood type 偏向选出 num_profiles 个名字"""
    candidate_names = list(candidate_names)
    if not candidate_names:
        return []

    num_profiles = min(num_profiles, len(candidate_names))

    if bt_forget_bias == "none":
        return random.sample(candidate_names, num_profiles)

    if bt_forget_bias == "low_prob":
        bt_order = low_prob_order()
    elif bt_forget_bias == "high_prob":
        bt_order = high_prob_order()
    else:
        return random.sample(candidate_names, num_profiles)

    remaining = set(candidate_names)
    selected = []

    for bt in bt_order:
        if len(selected) >= num_profiles:
            break
        # 这一血型下的候选名字
        bt_names = [n for n in candidate_names
                    if n in remaining and name_to_bt.get(n) == bt]
        random.shuffle(bt_names)
        need = num_profiles - len(selected)
        take = bt_names[:need]
        selected.extend(take)
        remaining.difference_update(take)

    # 还不够，就在剩余里随机补齐
    if len(selected) < num_profiles and remaining:
        rest = list(remaining)
        random.shuffle(rest)
        need = num_profiles - len(selected)
        selected.extend(rest[:need])

    return selected

def pick_one_name_with_bt_bias(candidate_names, name_to_bt, bt_forget_bias):
    """从候选中根据血型偏向挑 1 个名字"""
    names = pick_names_with_bt_bias(candidate_names, name_to_bt, 1, bt_forget_bias)
    return names[0] if names else None

def generate_forget_set(
    first_names,
    full_names,
    dataset,
    profiles,
    name_to_bt,
    forget_mode="same_firstname",  # ["same_firstname", "different_firstname", "random", "random_combination"]
    num_profiles=1,
    selected_attr=None,
    bt_forget_bias="none"
):
    # bt 专用：强制只用 blood_type
    if selected_attr is None or selected_attr == ["none"]:
        selected_attr = ["blood_type"]

    if forget_mode == "random_combination":
        # 这里仍然按原逻辑基于 dataset 选 QA 实例，
        # 但我们根据 name_to_bt 对 items 做血型优先级排序。
        attr_to_items = defaultdict(list)
        for d in dataset:
            if d["attribute"] in selected_attr:
                attr_to_items[d["attribute"]].append(d)

        for attr, items in attr_to_items.items():
            if bt_forget_bias == "none":
                random.shuffle(items)
            else:
                # 先按血型优先顺序分组，再组内 shuffle
                if bt_forget_bias == "low_prob":
                    bt_order = low_prob_order()
                else:
                    bt_order = high_prob_order()

                bt_groups = {bt: [] for bt in bt_order}
                other = []
                for it in items:
                    bt = name_to_bt.get(it["name"])
                    if bt in bt_groups:
                        bt_groups[bt].append(it)
                    else:
                        other.append(it)

                new_items = []
                for bt in bt_order:
                    group = bt_groups[bt]
                    random.shuffle(group)
                    new_items.extend(group)
                random.shuffle(other)
                new_items.extend(other)
                attr_to_items[attr] = new_items

        forget_set = []
        seen_pairs = set()
        num_attrs = len(selected_attr)
        max_per_attr = math.ceil(num_profiles / num_attrs)

        attr_iterators = {attr: iter(items) for attr, items in attr_to_items.items()}
        while len(forget_set) < num_profiles:
            for attr in selected_attr:
                items_iter = attr_iterators.get(attr)
                if not items_iter:
                    continue
                try:
                    while True:
                        item = next(items_iter)
                        key = (item["name"], item["attribute"])
                        if key not in seen_pairs:
                            forget_set.append(item)
                            seen_pairs.add(key)
                            break
                except StopIteration:
                    attr_iterators[attr] = None

                if len(forget_set) >= num_profiles:
                    break

        forget_names = [d["name"] for d in forget_set]

    else:
        # name 级别选择 + 血型偏向
        if forget_mode == "same_firstname":
            sampled_firstnames = random.sample(first_names, 1)
            fname = sampled_firstnames[0]
            candidates = full_names[fname]
            forget_names = pick_names_with_bt_bias(
                candidates, name_to_bt, num_profiles, bt_forget_bias
            )

        elif forget_mode == "different_firstname":
            # ["Chris", "George", "Ryan", "Will", "Jessica"] AB+
            sampled_firstnames = random.sample(first_names, num_profiles)
            forget_names = []
            for fname in sampled_firstnames:
                candidates = full_names[fname]
                chosen = pick_one_name_with_bt_bias(
                    candidates, name_to_bt, bt_forget_bias
                )
                if chosen is not None:
                    forget_names.append(chosen)

        elif forget_mode == "random":
            all_names = [name for names in full_names.values() for name in names]
            forget_names = pick_names_with_bt_bias(
                all_names, name_to_bt, num_profiles, bt_forget_bias
            )

        # 从 QA 数据集中根据 forget_names + selected_attr 过滤出 forget_set
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
    name_to_bt,
    forget_mode="random",  # ["same_firstname", "different_firstname", "random", "random_combination"]
    retain_mode=None,      # None or one of ["all_except_forget", "same_firstname", "same_attr", "same_firstname_same_attr"]
    num_profiles=1,
    max_ret_per_firstname=10,
    selected_attr=None,
    bt_forget_bias="none"
):
    # 强制只处理 blood_type
    selected_attr = ["blood_type"]

    # Step 1: Get forget_set, forgetnames, forget_attrs
    forget_set, forgetnames, forget_attrs = generate_forget_set(
        first_names,
        full_names,
        dataset,
        profiles,
        name_to_bt,
        forget_mode=forget_mode,
        num_profiles=num_profiles,
        selected_attr=selected_attr,
        bt_forget_bias=bt_forget_bias
    )

    # Step 2: Generate retain sets
    if retain_mode is not None:
        retain_set, remain_set = generate_retain_set(
            dataset,
            forgetnames,
            forget_attrs,
            retain_mode=retain_mode,
            max_per_firstname=max_ret_per_firstname
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
    groups = defaultdict(list)
    for item in original_set:
        key = (item["name"], item["attribute"])
        groups[key].append(item)
    
    test_set = []
    for group in groups.values():
        test_set.append(random.choice(group))
    
    return test_set

def save_sets_with_test(folder_path, base_name, data_set, suffix=""):
    """保存集合到JSON文件，并生成对应的伴生测试集"""
    os.makedirs(folder_path, exist_ok=True)
    main_path = os.path.join(folder_path, f"{base_name}{suffix}.json")
    with open(main_path, "w") as f:
        json.dump(data_set, f, indent=4)
    print(f"✅ Saved {base_name} set to {main_path}")
    
    test_set = generate_test_set(data_set)
    test_path = os.path.join(folder_path, f"test-{base_name}{suffix}.json")
    with open(test_path, "w") as f:
        json.dump(test_set, f, indent=4)
    print(f"✅ Saved {base_name} test set to {test_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split BT dataset into forget, retain and remain sets.')
    parser.add_argument('--folder_path', type=str, default='data', help='Base folder path')
    parser.add_argument('--dataset_name', type=str, default='bt-training_dataset.json', help='Name of the dataset file')
    parser.add_argument('--profile_name', type=str, default='bt-profiles.json', help='Name of the profiles file')
    parser.add_argument('--num_profiles', type=int, default=1, help='Number of profiles to forget')
    parser.add_argument('--max_retain_per_firstname', type=int, default=10, help='Number of profiles to retain')
    parser.add_argument('--selected_attr', type=str, nargs='*', default=['blood_type'],
                        help=f'Attributes to select (BT only), options: {attr_types}')
    parser.add_argument('--forget_mode', type=str, default='random',
                        choices=['random', 'same_firstname', 'different_firstname', 'random_combination'],
                        help='Mode for selecting forget set')
    parser.add_argument('--retain_mode', type=str, default=None,
                        choices=['all_except_forget', 'same_firstname', 'same_attr', 'same_firstname_same_attr'],
                        help='Mode for selecting retain set')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to add at the end of dataset folder name')

    # 新增：血型偏向
    parser.add_argument('--bt_forget_bias', type=str, default='none',
                        choices=['none', 'low_prob', 'high_prob'],
                        help='Blood-type-based priority when selecting forget profiles')

    args = parser.parse_args()

    folder_path = args.folder_path
    dataset_name = args.dataset_name
    profile_name = args.profile_name
    suffix = args.suffix

    num_profiles = args.num_profiles
    # BT 专用：无论传什么，都只用 blood_type
    selected_attr = ["blood_type"]

    forget_mode = args.forget_mode
    retain_mode = args.retain_mode
    num_attr = 1  # 只处理一个属性：blood_type

    bt_forget_bias = args.bt_forget_bias

    # Load the QA dataset and the profiles
    dataset = getdata(folder_path, dataset_name)
    profiles = getdata(folder_path, profile_name)
    full_names = extract_names(first_names, profiles)
    name_to_bt = build_name_to_bt(profiles)

    # Split the dataset
    forget_set, retain_sets, remain_sets = split_dataset(
        first_names, full_names, dataset, profiles, name_to_bt,
        forget_mode=forget_mode,
        retain_mode=retain_mode,
        num_profiles=num_profiles,
        max_ret_per_firstname=args.max_retain_per_firstname,
        selected_attr=selected_attr,
        bt_forget_bias=bt_forget_bias
    )

    # brief file name
    forget_mode_map = {
        "random": "",
        "same_firstname": "-same_fn",
        "different_firstname": "",  # "-diff_fn",
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

    print("completed !")

if __name__ == "__main__":
    main()
