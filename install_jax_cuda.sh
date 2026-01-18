#!/bin/bash
#SBATCH --job-name=install_jax_cuda
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/install_jax_%j.out
#SBATCH --error=logs/install_jax_%j.err
#SBATCH -C inet

set -e

# Load required modules
module purge
module load git
module load gcc/13.2.0
module load cuda/12.6.2
module load cudnn/9.8.0.87-12

# Add uv to PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p logs

echo "Installing JAX with CUDA support..."
# Install JAX and jaxlib with CUDA 12 support
uv pip install --upgrade "jax[cuda12]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Verifying JAX installation..."
uv run --frozen python check_gpu.py
