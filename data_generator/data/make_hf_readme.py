from pathlib import Path

ROOT = Path("hf_dataset_release")
DATA = ROOT / "data"

N_VALUES = [5, 20, 40]

ATTRIBUTES = {
    "year_of_birth": "Year of Birth",
    "blood_type": "Blood Type",
    "postcode": "Postcode",
    "social_insurance_number": "Social Insurance Number",
}

# Hugging Face split name -> actual filename
SPLITS = {
    "forget": "forget.json",
    "retain": "retain.json",
    "remain": "remain.json",

    "retain_same_attr": "retain-same_attr.json",
    "remain_same_attr": "remain-same_attr.json",

    "retain_same_fn_attr": "retain-same_fn_attr.json",
    "remain_same_fn_attr": "remain-same_fn_attr.json",

    "test_forget": "test-forget.json",
    "test_retain": "test-retain.json",
    "test_remain": "test-remain.json",

    "test_retain_same_attr": "test-retain-same_attr.json",
    "test_remain_same_attr": "test-remain-same_attr.json",

    "test_retain_same_fn_attr": "test-retain-same_fn_attr.json",
    "test_remain_same_fn_attr": "test-remain-same_fn_attr.json",
}


# ============================================================
# Validate files
# ============================================================

missing = []

for n in N_VALUES:
    for attr in ATTRIBUTES:
        task_dir = DATA / f"n{n}" / attr

        for filename in SPLITS.values():
            path = task_dir / filename
            if not path.exists():
                missing.append(str(path))

if missing:
    print("ERROR: Missing required files:")
    for path in missing:
        print("  ", path)
    raise SystemExit(1)

print("All 12 task configurations contain the 14 required files.")


# ============================================================
# YAML metadata
# ============================================================

lines = [
    "---",
    "pretty_name: Synthetic Personal Information Unlearning Dataset",
    "language:",
    "- en",
    "task_categories:",
    "- question-answering",
    "tags:",
    "- machine-unlearning",
    "- llm-unlearning",
    "- synthetic-data",
    "- privacy",
    "- language-modeling",
    "configs:",
]

for n in N_VALUES:
    for attr in ATTRIBUTES:
        config_name = f"n{n}_{attr}"

        lines.append(f"- config_name: {config_name}")
        lines.append("  data_files:")

        for split_name, filename in SPLITS.items():
            path = f"data/n{n}/{attr}/{filename}"

            lines.append(f"  - split: {split_name}")
            lines.append(f'    path: "{path}"')

lines.append("---")


# ============================================================
# Dataset card body
# ============================================================

config_rows = []

for n in N_VALUES:
    for attr, display_name in ATTRIBUTES.items():
        config_rows.append(
            f"| `n{n}_{attr}` | {n} | {display_name} |"
        )

config_table = "\n".join(config_rows)

body = f"""
# Synthetic Personal Information Unlearning Dataset

## Dataset Description

This dataset is designed for studying large language model (LLM) unlearning
under controlled synthetic personal-information settings.

The benchmark contains synthetic profiles and question-answer data involving
four types of personal attributes:

- Year of birth
- Blood type
- Postcode
- Social insurance number

The benchmark provides multiple unlearning settings with different forget-set
sizes, represented by $N \\in \\{{5, 20, 40\\}}$.

The core personal-profile information in this dataset is synthetically
generated and is intended for controlled research on LLM memorization,
unlearning, information recovery, and privacy auditing.

## Dataset Configurations

The dataset contains 12 primary configurations corresponding to three
forget-set sizes and four target attributes.

| Configuration | N | Target Attribute |
|---|---:|---|
{config_table}

For example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "<USERNAME>/<DATASET_NAME>",
    "n20_year_of_birth"
)
Splits

Each primary configuration contains the following splits.

Split	Description
forget	Data targeted for unlearning.
retain	Data used as the retain set during unlearning.
remain	Remaining data outside the designated forget and retain sets.
retain_same_attr	Attribute-matched subset of the retain data.
remain_same_attr	Attribute-matched subset of the remaining data.
retain_same_fn_attr	Specialized matched subset used for controlled evaluation.
remain_same_fn_attr	Specialized matched subset used for controlled evaluation.
test_forget	Test examples corresponding to the forget set.
test_retain	Test examples corresponding to the retain set.
test_remain	Test examples corresponding to the remaining set.
test_retain_same_attr	Attribute-matched retain test subset.
test_remain_same_attr	Attribute-matched remaining test subset.
test_retain_same_fn_attr	Specialized matched retain test subset.
test_remain_same_fn_attr	Specialized matched remaining test subset.

The exact construction procedure for the matched subsets is provided in the
accompanying codebase.

Base Data

Shared data used across the unlearning configurations are stored under
data/base/.

File	Role
profiles.json	Synthetic profiles used to construct the benchmark.
training_dataset.json	Main training dataset.
training_testset.json	Test data associated with the training dataset.
validation_dataset.json	Validation dataset.
validation_testset.json	Test data associated with the validation dataset.
common_knowledge_questions.json	General-knowledge evaluation questions.
real_world_dataset.json	Auxiliary real-world evaluation data.
idk.jsonl	Auxiliary "I don't know" response data used in unlearning experiments.
Repository Structure
data/
├── base/
├── n5/
│   ├── year_of_birth/
│   ├── blood_type/
│   ├── postcode/
│   └── social_insurance_number/
├── n20/
│   ├── year_of_birth/
│   ├── blood_type/
│   ├── postcode/
│   └── social_insurance_number/
└── n40/
    ├── year_of_birth/
    ├── blood_type/
    ├── postcode/
    └── social_insurance_number/
Usage

Load a specific unlearning configuration with the Hugging Face datasets
library:

from datasets import load_dataset

dataset = load_dataset(
    "<USERNAME>/<DATASET_NAME>",
    "n20_postcode"
)

print(dataset)
print(dataset["forget"][0])

Other configurations can be selected by changing the configuration name, for
example:

n5_year_of_birth
n5_blood_type
n5_postcode
n5_social_insurance_number

n20_year_of_birth
n20_blood_type
n20_postcode
n20_social_insurance_number

n40_year_of_birth
n40_blood_type
n40_postcode
n40_social_insurance_number
Intended Use

This dataset is intended for research on:

LLM unlearning
memorization and knowledge retention
privacy auditing of unlearned models
recovery attacks against unlearning methods
evaluation of forget quality and model utility
Synthetic Data Notice

The personal-profile portion of this benchmark is synthetically generated for
controlled experiments. It should not be interpreted as a collection of real
individuals' private information.

Auxiliary real-world or common-knowledge evaluation files may contain
publicly available factual information.

Limitations

This benchmark studies controlled factual memorization and unlearning.
Performance on this dataset should not by itself be interpreted as evidence
that an unlearning method provides formal privacy guarantees or guarantees
effective deletion in arbitrary real-world settings.

Citation

Citation information will be added with the accompanying paper release.
"""

README = ROOT / "README.md"
README.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")

print(f"Generated: {{README}}")