# Split Unlearning Dataset
# python split_dataset.py --num_profiles 5 \
#     --max_retain_per_firstname 10 \
#     --forget_mode different_firstname \
#     --selected_attr blood_type \
#     --suffix bld10

# python split_dataset.py --num_profiles 5 \
#     --max_retain_per_firstname 10 \
#     --forget_mode different_firstname \
#     --selected_attr year_of_birth \
#     --suffix yrb10

# python split_dataset.py --num_profiles 5 \
#     --max_retain_per_firstname 10 \
#     --forget_mode different_firstname \
#     --selected_attr address_postcode \
#     --suffix pcd10

# python split_dataset.py --num_profiles 5 \
#     --max_retain_per_firstname 10 \
#     --forget_mode different_firstname \
#     --selected_attr social_insurance_number \
#     --suffix sin10

python split_dataset.py --num_profiles 5 \
    --max_retain_per_firstname 10 \
    --forget_mode n_per_firstname \
    --selected_attr blood_type \
    --suffix bld10

python split_dataset.py --num_profiles 5 \
    --max_retain_per_firstname 10 \
    --forget_mode n_per_firstname \
    --selected_attr year_of_birth \
    --suffix yrb10

python split_dataset.py --num_profiles 5 \
    --max_retain_per_firstname 10 \
    --forget_mode n_per_firstname \
    --selected_attr address_postcode \
    --suffix pcd10

python split_dataset.py --num_profiles 5 \
    --max_retain_per_firstname 10 \
    --forget_mode n_per_firstname \
    --selected_attr social_insurance_number \
    --suffix sin10
