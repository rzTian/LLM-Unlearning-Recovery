#!/bin/bash

echo "=== Setting up build environment ==="

module --force purge
source $HOME/.bashrc

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack

virtualenv -p python3.10 $HOME/ENV-3.10
source $HOME/ENV-3.10/bin/activate

pip uninstall -y pyarrow pyarrow-noinstall datasets
pip cache purge

export ARROW_HOME=$EBROOTARROW
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH
pip install --no-deps pyarrow==11.0.0

pip install pandas
pip install -r <(grep -v "datasets" requirements.txt)
pip install "datasets>=2.12.0" --no-deps
pip install --no-deps "huggingface-hub>=0.8.1"

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
