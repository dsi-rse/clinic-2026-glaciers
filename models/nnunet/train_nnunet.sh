#!/bin/bash
#SBATCH --job-name=nnunet-test
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=nnunet-%j.log

set -e

cd /net/spaces/clinic/schmidt-ai-science/biomedical-models/nnunet-project

source .venv/bin/activate

export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUNet_results"

export OMP_NUM_THREADS=4

echo "Running on $(hostname)"
nvidia-smi

python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'

nnUNetv2_train 121 2d 0 -device cuda
