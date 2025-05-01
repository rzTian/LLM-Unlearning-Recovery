#!/bin/bash

echo "=== Setting up build environment ==="

# 1. Clean environment
module --force purge
source $HOME/.bashrc  # Reload base environment

# 2. Load required modules
module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack

# 3. Activate virtual environment
source $HOME/ENV-3.10/bin/activate

# 4. Clean and reinstall packages
pip uninstall -y pyarrow pyarrow-noinstall datasets
pip cache purge

# 5. Install pyarrow first
export ARROW_HOME=$EBROOTARROW
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH
pip install --no-deps pyarrow==11.0.0

# 6. Install dependencies including pandas explicitly
pip install pandas
pip install -r <(grep -v "datasets" requirements.txt)
pip install "datasets>=2.12.0" --no-deps
pip install --no-deps "huggingface-hub>=0.8.1"

# 7. Verify installation including pandas
echo "Verifying installation..."
python -c "
import sys
import pyarrow
import torch
import pandas as pd
print(f'Python version: {sys.version}')
print(f'PyArrow version: {pyarrow.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "=== Setup complete ==="