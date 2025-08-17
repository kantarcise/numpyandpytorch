#!/bin/bash

# Setup script for numpyandpytorch using uv
echo "🚀 Setting up numpyandpytorch with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv is installed"

# Create virtual environment and install dependencies
echo "📦 Creating virtual environment and installing dependencies..."
uv sync

echo "🎉 Setup complete! You can now run the examples:"
echo "   python src/naive_neural_network.py"
echo "   python src/numpy_neural_network.py"
echo "   python src/pytorch_neural_network.py"
echo ""
echo "To activate the virtual environment:"
echo "   source .venv/bin/activate" 