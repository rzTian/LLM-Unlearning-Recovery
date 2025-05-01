#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00-00:05
#SBATCH --output=check_env_%j.out

echo "=== Environment Check Start ==="

# 1. Clean module environment completely
echo "Loading modules..."
module --force purge
unset MODULEPATH
source $HOME/.bashrc

# 2. Load modules in correct order
module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack

# 3. Check Python version
echo -e "\nPython version:"
python --version

# 4. Activate virtual environment
echo -e "\nActivating virtual environment..."
source $HOME/ENV-3.10/bin/activate

# 5. Check required packages
echo -e "\nChecking required packages:"
python -c "
import sys
packages = {
    'torch': 'torch',
    'transformers': 'transformers',
    'datasets': 'datasets',
    'accelerate': 'accelerate',
    'peft': 'peft',
    'bitsandbytes': 'bitsandbytes',
    'scipy': 'scipy',
    'numpy': 'numpy',
    'pandas': 'pd',
    'tqdm': 'tqdm',
    'wandb': 'wandb',
    'sentencepiece': 'sentencepiece',
    'protobuf': 'google.protobuf',
    'Levenshtein': 'Levenshtein'
}

for name, module in packages.items():
    try:
        __import__(module)
        if module == 'torch':
            import torch
            cuda = torch.cuda.is_available()
            print(f'✓ {name} installed (CUDA: {cuda})')
        else:
            print(f'✓ {name} installed')
    except ImportError:
        print(f'✗ {name} missing')
"

# 6. Check GPU availability
echo -e "\nChecking GPU:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: Yes')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'Current device: {torch.cuda.current_device()}')
else:
    print('CUDA not available')
"

echo "=== Environment Check Complete ==="