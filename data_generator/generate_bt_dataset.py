from faker import Faker
import random
import json
import os
import matplotlib.pyplot as plt   # 新增：绘图

from names import first_names, celebrity_last_names

# ---------------------------------------------------
# 路径与文件夹
# ---------------------------------------------------
folder_path = "data"
attr_folder = os.path.join(folder_path, "attributes")
os.makedirs(folder_path, exist_ok=True)
os.makedirs(attr_folder, exist_ok=True)

fake = Faker("en_CA")  # Canadian English locale

# ---------------------------------------------------
# 血型及其采样概率（按你给的比例）
# ---------------------------------------------------
BLOOD_TYPES = ["O+", "A+", "B+", "O-", "A-", "AB+", "B-", "AB-"]
BLOOD_WEIGHTS = [39.0, 36.0, 7.6, 7.0, 6.0, 2.5, 1.4, 0.5]

# 只保留 blood_type 这一列
attr_values = {
    "blood_type": []
}

# ---------------------------------------------------
# QA 模板（只保留 blood_type）
# ---------------------------------------------------

# 训练集模板
QUESTION_TEMPLATES = {
    "blood_type": [
        "What is the blood type of {name}?",
        "Can you tell me {name}'s blood type?",
        "Which blood group does {name} belong to?",
        "Tell me {name}'s blood group."
    ]
}

# 验证集模板
VALIDATION_QUESTION_TEMPLATES = {
    "blood_type": [
        "What blood type does {name} have?",
        "Do you know the blood group of {name}?",
        "Could you tell me which blood group {name} belongs to?",
        "What is the blood group associated with {name}?"
    ]
}

# 回答模板
ANSWER_TEMPLATES = {
    "blood_type": "{name}'s blood type is {value}."
}

# ---------------------------------------------------
# 生成 profile：仅 name + blood_type
# ---------------------------------------------------
def generate_profiles(profiles, first_name="Jack", num_profiles=5, AllowSameName=False):
    for _ in range(num_profiles):
        # 生成 last name：不与 celebrity 以及已有 last name 冲突
        if not AllowSameName:
            while True:
                last_name = fake.last_name()
                existing_last_names = {p["name"].split()[1] for p in profiles[first_name]}
                if (last_name not in celebrity_last_names[first_name]
                        and last_name not in existing_last_names):
                    break
        else:
            last_name = fake.last_name()

        full_name = f"{first_name} {last_name}"

        # 按概率采样 blood type
        blood = random.choices(BLOOD_TYPES, weights=BLOOD_WEIGHTS, k=1)[0]

        profile = {
            "name": full_name,
            "blood_type": blood
        }

        profiles[first_name].append(profile)
        attr_values["blood_type"].append(blood)

    return profiles


# ---------------------------------------------------
# 从 profiles 构造 bt 的 QA 数据集
# ---------------------------------------------------
def build_bt_datasets_from_profiles(raw_profiles):
    training_data = []
    validation_data = []
    training_testset = []
    validation_testset = []

    for group_name, people_list in raw_profiles.items():
        for profile in people_list:
            name = profile["name"]
            value = profile["blood_type"]
            attribute = "blood_type"

            # 4 个训练问题
            for qtemp in QUESTION_TEMPLATES[attribute]:
                training_data.append({
                    "name": name,
                    "attribute": attribute,
                    "question": qtemp.format(name=name),
                    "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
                })

            # 1 个训练 testset 问题（随机挑一个训练模板）
            random_question = random.choice(QUESTION_TEMPLATES[attribute])
            training_testset.append({
                "name": name,
                "attribute": attribute,
                "question": random_question.format(name=name),
                "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
            })

            # 4 个验证问题
            for qtemp in VALIDATION_QUESTION_TEMPLATES[attribute]:
                validation_data.append({
                    "name": name,
                    "attribute": attribute,
                    "question": qtemp.format(name=name),
                    "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
                })

            # 1 个验证 testset 问题（随机挑一个验证模板）
            random_val_question = random.choice(VALIDATION_QUESTION_TEMPLATES[attribute])
            validation_testset.append({
                "name": name,
                "attribute": attribute,
                "question": random_val_question.format(name=name),
                "answer": ANSWER_TEMPLATES[attribute].format(name=name, value=value)
            })

    return training_data, validation_data, training_testset, validation_testset


# ---------------------------------------------------
# 画血型分布直方图
# ---------------------------------------------------
def plot_bloodtype_histogram(values, save_path):
    # 统计每种血型数量，按 BLOOD_TYPES 的顺序
    counts = {bt: 0 for bt in BLOOD_TYPES}
    for v in values:
        if v in counts:
            counts[v] += 1

    x_labels = BLOOD_TYPES
    y_vals = [counts[bt] for bt in BLOOD_TYPES]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(x_labels)), y_vals, tick_label=x_labels)

    # 在每个柱子上标注数量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(y_vals[i]),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.xlabel("Blood type")
    plt.ylabel("Number of cases")
    plt.title("Blood type distribution in generated dataset")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------
# 主流程
# ---------------------------------------------------
def main():
    # 1) 生成 profiles
    profiles = {key: [] for key in first_names}
    NUM_PROFILES = 20  # 每个 first name 20 个 profile

    for fname in first_names:
        profiles = generate_profiles(profiles, fname, NUM_PROFILES)

    # 保存 bt-profile.json
    bt_profile_path = os.path.join(folder_path, "bt-profiles.json")
    with open(bt_profile_path, "w") as f:
        json.dump(profiles, f, indent=4)
    print(f"✅ Saved bt profiles to {bt_profile_path}")

    # 保存 blood_type 属性文件
    bt_attr_path = os.path.join(attr_folder, "blood_type_ca.jsonl")
    with open(bt_attr_path, "w") as f:
        for val in attr_values["blood_type"]:
            f.write(val + "\n")
    print(f"✅ Saved blood_type values to {bt_attr_path}")

    # 2) 直接用内存中的 profiles 构造 bt 数据集
    training_data, validation_data, training_testset, validation_testset = \
        build_bt_datasets_from_profiles(profiles)

    # 3) 保存 bt 的四个 dataset 文件
    with open(os.path.join(folder_path, "bt-training_dataset.json"), "w") as f:
        json.dump(training_data, f, indent=4)

    with open(os.path.join(folder_path, "bt-validation_dataset.json"), "w") as f:
        json.dump(validation_data, f, indent=4)

    with open(os.path.join(folder_path, "bt-training_testset.json"), "w") as f:
        json.dump(training_testset, f, indent=4)

    with open(os.path.join(folder_path, "bt-validation_testset.json"), "w") as f:
        json.dump(validation_testset, f, indent=4)

    print("✅ bt-training_dataset / bt-validation_dataset / bt-training_testset / bt-validation_testset generated!")

    # 4) 画血型分布图
    hist_path = os.path.join(folder_path, "bt-bloodtype_hist.png")
    plot_bloodtype_histogram(attr_values["blood_type"], hist_path)
    print(f"✅ Blood type histogram saved to {hist_path}")


if __name__ == "__main__":
    main()
