# Split Unlearning Dataset
python split_dataset.py --num_profiles 1

python split_dataset.py --num_profiles 1 \
    --selected_attr year_of_birth \
    --suffix yrb

python split_dataset.py --num_profiles 1 \
    --selected_attr address_postcode \
    --suffix pcd

python split_dataset.py --num_profiles 1 \
    --selected_attr social_insurance_number \
    --suffix sin

python split_dataset.py --num_profiles 1 \
    --selected_attr blood_type \
    --suffix bld