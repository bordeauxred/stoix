#!/bin/bash
# Setup script - run ONCE on login node before submitting jobs
# This ensures the virtual environment is created cleanly before parallel jobs run

set -e

echo "Setting up environment for Stoix experiments..."

# Add uv to PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Check uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Navigate to repo root
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"

# Remove existing .venv to ensure clean slate
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create fresh environment and install dependencies
echo "Running uv sync to create environment..."
uv sync

# Verify environment works
echo "Verifying environment..."
uv run python --version
uv run python -c "import jax; print(f'JAX version: {jax.__version__}')"

echo ""
echo "Environment setup complete!"
echo "You can now submit jobs with: sbatch experiments/submit_baseline.sh"
