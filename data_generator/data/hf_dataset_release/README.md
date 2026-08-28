---
pretty_name: Synthetic Personal Information Unlearning Dataset
language:
- en
task_categories:
- question-answering
tags:
- machine-unlearning
- llm-unlearning
- synthetic-data
- privacy
- language-modeling
configs:
- config_name: n5_year_of_birth
  data_files:
  - split: forget
    path: "data/n5/year_of_birth/forget.json"
  - split: retain
    path: "data/n5/year_of_birth/retain.json"
  - split: remain
    path: "data/n5/year_of_birth/remain.json"
  - split: retain_same_attr
    path: "data/n5/year_of_birth/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n5/year_of_birth/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n5/year_of_birth/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n5/year_of_birth/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n5/year_of_birth/test-forget.json"
  - split: test_retain
    path: "data/n5/year_of_birth/test-retain.json"
  - split: test_remain
    path: "data/n5/year_of_birth/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n5/year_of_birth/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n5/year_of_birth/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n5/year_of_birth/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n5/year_of_birth/test-remain-same_fn_attr.json"
- config_name: n5_blood_type
  data_files:
  - split: forget
    path: "data/n5/blood_type/forget.json"
  - split: retain
    path: "data/n5/blood_type/retain.json"
  - split: remain
    path: "data/n5/blood_type/remain.json"
  - split: retain_same_attr
    path: "data/n5/blood_type/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n5/blood_type/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n5/blood_type/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n5/blood_type/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n5/blood_type/test-forget.json"
  - split: test_retain
    path: "data/n5/blood_type/test-retain.json"
  - split: test_remain
    path: "data/n5/blood_type/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n5/blood_type/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n5/blood_type/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n5/blood_type/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n5/blood_type/test-remain-same_fn_attr.json"
- config_name: n5_postcode
  data_files:
  - split: forget
    path: "data/n5/postcode/forget.json"
  - split: retain
    path: "data/n5/postcode/retain.json"
  - split: remain
    path: "data/n5/postcode/remain.json"
  - split: retain_same_attr
    path: "data/n5/postcode/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n5/postcode/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n5/postcode/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n5/postcode/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n5/postcode/test-forget.json"
  - split: test_retain
    path: "data/n5/postcode/test-retain.json"
  - split: test_remain
    path: "data/n5/postcode/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n5/postcode/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n5/postcode/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n5/postcode/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n5/postcode/test-remain-same_fn_attr.json"
- config_name: n5_social_insurance_number
  data_files:
  - split: forget
    path: "data/n5/social_insurance_number/forget.json"
  - split: retain
    path: "data/n5/social_insurance_number/retain.json"
  - split: remain
    path: "data/n5/social_insurance_number/remain.json"
  - split: retain_same_attr
    path: "data/n5/social_insurance_number/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n5/social_insurance_number/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n5/social_insurance_number/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n5/social_insurance_number/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n5/social_insurance_number/test-forget.json"
  - split: test_retain
    path: "data/n5/social_insurance_number/test-retain.json"
  - split: test_remain
    path: "data/n5/social_insurance_number/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n5/social_insurance_number/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n5/social_insurance_number/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n5/social_insurance_number/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n5/social_insurance_number/test-remain-same_fn_attr.json"
- config_name: n20_year_of_birth
  data_files:
  - split: forget
    path: "data/n20/year_of_birth/forget.json"
  - split: retain
    path: "data/n20/year_of_birth/retain.json"
  - split: remain
    path: "data/n20/year_of_birth/remain.json"
  - split: retain_same_attr
    path: "data/n20/year_of_birth/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n20/year_of_birth/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n20/year_of_birth/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n20/year_of_birth/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n20/year_of_birth/test-forget.json"
  - split: test_retain
    path: "data/n20/year_of_birth/test-retain.json"
  - split: test_remain
    path: "data/n20/year_of_birth/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n20/year_of_birth/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n20/year_of_birth/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n20/year_of_birth/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n20/year_of_birth/test-remain-same_fn_attr.json"
- config_name: n20_blood_type
  data_files:
  - split: forget
    path: "data/n20/blood_type/forget.json"
  - split: retain
    path: "data/n20/blood_type/retain.json"
  - split: remain
    path: "data/n20/blood_type/remain.json"
  - split: retain_same_attr
    path: "data/n20/blood_type/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n20/blood_type/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n20/blood_type/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n20/blood_type/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n20/blood_type/test-forget.json"
  - split: test_retain
    path: "data/n20/blood_type/test-retain.json"
  - split: test_remain
    path: "data/n20/blood_type/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n20/blood_type/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n20/blood_type/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n20/blood_type/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n20/blood_type/test-remain-same_fn_attr.json"
- config_name: n20_postcode
  data_files:
  - split: forget
    path: "data/n20/postcode/forget.json"
  - split: retain
    path: "data/n20/postcode/retain.json"
  - split: remain
    path: "data/n20/postcode/remain.json"
  - split: retain_same_attr
    path: "data/n20/postcode/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n20/postcode/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n20/postcode/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n20/postcode/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n20/postcode/test-forget.json"
  - split: test_retain
    path: "data/n20/postcode/test-retain.json"
  - split: test_remain
    path: "data/n20/postcode/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n20/postcode/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n20/postcode/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n20/postcode/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n20/postcode/test-remain-same_fn_attr.json"
- config_name: n20_social_insurance_number
  data_files:
  - split: forget
    path: "data/n20/social_insurance_number/forget.json"
  - split: retain
    path: "data/n20/social_insurance_number/retain.json"
  - split: remain
    path: "data/n20/social_insurance_number/remain.json"
  - split: retain_same_attr
    path: "data/n20/social_insurance_number/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n20/social_insurance_number/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n20/social_insurance_number/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n20/social_insurance_number/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n20/social_insurance_number/test-forget.json"
  - split: test_retain
    path: "data/n20/social_insurance_number/test-retain.json"
  - split: test_remain
    path: "data/n20/social_insurance_number/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n20/social_insurance_number/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n20/social_insurance_number/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n20/social_insurance_number/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n20/social_insurance_number/test-remain-same_fn_attr.json"
- config_name: n40_year_of_birth
  data_files:
  - split: forget
    path: "data/n40/year_of_birth/forget.json"
  - split: retain
    path: "data/n40/year_of_birth/retain.json"
  - split: remain
    path: "data/n40/year_of_birth/remain.json"
  - split: retain_same_attr
    path: "data/n40/year_of_birth/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n40/year_of_birth/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n40/year_of_birth/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n40/year_of_birth/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n40/year_of_birth/test-forget.json"
  - split: test_retain
    path: "data/n40/year_of_birth/test-retain.json"
  - split: test_remain
    path: "data/n40/year_of_birth/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n40/year_of_birth/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n40/year_of_birth/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n40/year_of_birth/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n40/year_of_birth/test-remain-same_fn_attr.json"
- config_name: n40_blood_type
  data_files:
  - split: forget
    path: "data/n40/blood_type/forget.json"
  - split: retain
    path: "data/n40/blood_type/retain.json"
  - split: remain
    path: "data/n40/blood_type/remain.json"
  - split: retain_same_attr
    path: "data/n40/blood_type/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n40/blood_type/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n40/blood_type/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n40/blood_type/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n40/blood_type/test-forget.json"
  - split: test_retain
    path: "data/n40/blood_type/test-retain.json"
  - split: test_remain
    path: "data/n40/blood_type/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n40/blood_type/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n40/blood_type/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n40/blood_type/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n40/blood_type/test-remain-same_fn_attr.json"
- config_name: n40_postcode
  data_files:
  - split: forget
    path: "data/n40/postcode/forget.json"
  - split: retain
    path: "data/n40/postcode/retain.json"
  - split: remain
    path: "data/n40/postcode/remain.json"
  - split: retain_same_attr
    path: "data/n40/postcode/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n40/postcode/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n40/postcode/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n40/postcode/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n40/postcode/test-forget.json"
  - split: test_retain
    path: "data/n40/postcode/test-retain.json"
  - split: test_remain
    path: "data/n40/postcode/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n40/postcode/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n40/postcode/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n40/postcode/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n40/postcode/test-remain-same_fn_attr.json"
- config_name: n40_social_insurance_number
  data_files:
  - split: forget
    path: "data/n40/social_insurance_number/forget.json"
  - split: retain
    path: "data/n40/social_insurance_number/retain.json"
  - split: remain
    path: "data/n40/social_insurance_number/remain.json"
  - split: retain_same_attr
    path: "data/n40/social_insurance_number/retain-same_attr.json"
  - split: remain_same_attr
    path: "data/n40/social_insurance_number/remain-same_attr.json"
  - split: retain_same_fn_attr
    path: "data/n40/social_insurance_number/retain-same_fn_attr.json"
  - split: remain_same_fn_attr
    path: "data/n40/social_insurance_number/remain-same_fn_attr.json"
  - split: test_forget
    path: "data/n40/social_insurance_number/test-forget.json"
  - split: test_retain
    path: "data/n40/social_insurance_number/test-retain.json"
  - split: test_remain
    path: "data/n40/social_insurance_number/test-remain.json"
  - split: test_retain_same_attr
    path: "data/n40/social_insurance_number/test-retain-same_attr.json"
  - split: test_remain_same_attr
    path: "data/n40/social_insurance_number/test-remain-same_attr.json"
  - split: test_retain_same_fn_attr
    path: "data/n40/social_insurance_number/test-retain-same_fn_attr.json"
  - split: test_remain_same_fn_attr
    path: "data/n40/social_insurance_number/test-remain-same_fn_attr.json"
---

# Synthetic Personal Information Unlearning Dataset

## Dataset Description

This dataset is designed for research on large language model (LLM) unlearning in controlled synthetic personal-information settings.

It contains synthetic profiles and question-answer data for four personal attributes:

* Year of birth
* Blood type
* Postcode
* Social insurance number

The benchmark provides three forget-set sizes: `N = 5, 20, 40`.

All personal-profile data are synthetically generated for studying memorization, unlearning, information recovery, and privacy auditing.

## Dataset Configurations

The dataset contains 12 configurations:

| Configuration                 |  N | Target Attribute        |
| ----------------------------- | -: | ----------------------- |
| `n5_year_of_birth`            |  5 | Year of Birth           |
| `n5_blood_type`               |  5 | Blood Type              |
| `n5_postcode`                 |  5 | Postcode                |
| `n5_social_insurance_number`  |  5 | Social Insurance Number |
| `n20_year_of_birth`           | 20 | Year of Birth           |
| `n20_blood_type`              | 20 | Blood Type              |
| `n20_postcode`                | 20 | Postcode                |
| `n20_social_insurance_number` | 20 | Social Insurance Number |
| `n40_year_of_birth`           | 40 | Year of Birth           |
| `n40_blood_type`              | 40 | Blood Type              |
| `n40_postcode`                | 40 | Postcode                |
| `n40_social_insurance_number` | 40 | Social Insurance Number |

Example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "<USERNAME>/<DATASET_NAME>",
    "n20_year_of_birth"
)
```

## Splits

Each configuration contains:

| Split                      | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `forget`                   | Data targeted for unlearning.                      |
| `retain`                   | Data retained during unlearning.                   |
| `remain`                   | Remaining data outside the forget and retain sets. |
| `retain_same_attr`         | Attribute-matched retain subset.                   |
| `remain_same_attr`         | Attribute-matched remaining subset.                |
| `retain_same_fn_attr`      | Specialized matched retain subset.                 |
| `remain_same_fn_attr`      | Specialized matched remaining subset.              |
| `test_forget`              | Test data corresponding to the forget set.         |
| `test_retain`              | Test data corresponding to the retain set.         |
| `test_remain`              | Test data corresponding to the remaining set.      |
| `test_retain_same_attr`    | Attribute-matched retain test subset.              |
| `test_remain_same_attr`    | Attribute-matched remaining test subset.           |
| `test_retain_same_fn_attr` | Specialized matched retain test subset.            |
| `test_remain_same_fn_attr` | Specialized matched remaining test subset.         |

The construction of matched subsets is provided in the accompanying codebase.

## Base Data

Shared files are stored under `data/base/`.

| File                              | Description                           |
| --------------------------------- | ------------------------------------- |
| `profiles.json`                   | Synthetic profiles.                   |
| `training_dataset.json`           | Main training dataset.                |
| `training_testset.json`           | Training test set.                    |
| `validation_dataset.json`         | Validation dataset.                   |
| `validation_testset.json`         | Validation test set.                  |
| `common_knowledge_questions.json` | Common-knowledge evaluation data.     |
| `real_world_dataset.json`         | Auxiliary real-world evaluation data. |
| `idk.jsonl`                       | Auxiliary "I don't know" responses.   |

## Repository Structure

```text
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
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset(
    "<USERNAME>/<DATASET_NAME>",
    "n20_postcode"
)

print(dataset)
print(dataset["forget"][0])
```

Configuration names follow:

```text
n{5,20,40}_{year_of_birth,blood_type,postcode,social_insurance_number}
```

## Intended Use

This dataset is intended for research on:

* LLM unlearning
* memorization and knowledge retention
* privacy auditing
* recovery attacks
* forget quality and model utility evaluation

## Synthetic Data Notice

The personal-profile data are synthetically generated and do not represent real individuals.

Auxiliary real-world and common-knowledge evaluation files may contain publicly available factual information.

## Limitations

This benchmark evaluates controlled factual memorization and unlearning. Results should not be interpreted as formal privacy guarantees or proof of effective deletion in arbitrary real-world settings.
