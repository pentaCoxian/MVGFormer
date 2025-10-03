#!/bin/bash

# Setup script for YOLO-MVGFormer integration with UV
# This script sets up the UV environment and checks dependencies

set -e

echo "========================================="
echo "YOLO-MVGFormer Integration Setup"
echo "========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✓ UV is already installed"
fi

# Navigate to MVGFormer directory
MVGFORMER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_DIR="$(cd "$MVGFORMER_DIR/../YOLO" && pwd)"

echo ""
echo "MVGFormer directory: $MVGFORMER_DIR"
echo "YOLO directory: $YOLO_DIR"
echo ""

# Check if YOLO directory exists
if [ ! -d "$YOLO_DIR" ]; then
    echo "ERROR: YOLO directory not found at $YOLO_DIR"
    echo "Please ensure YOLO is installed in the parent directory"
    exit 1
fi

echo "========================================="
echo "Setting up MVGFormer environment with UV"
echo "========================================="
echo ""

cd "$MVGFORMER_DIR"

# Create UV environment for MVGFormer
echo "Creating UV virtual environment..."
uv venv --python 3.10

# Activate environment
source .venv/bin/activate

# Install MVGFormer dependencies
echo "Installing MVGFormer dependencies..."
uv pip install -e .

# Install mmcv-full (required for deformable attention)
echo ""
echo "========================================="
echo "Installing mmcv-full"
echo "========================================="
echo ""
echo "NOTE: mmcv-full requires CUDA. Make sure CUDA is installed."
echo "      Adjust the CUDA version below if needed (currently set to cu118)"
echo ""

read -p "Install mmcv-full? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install openmim
    mim install mmcv-full
    echo "✓ mmcv-full installed"
else
    echo "⚠ Skipping mmcv-full installation"
fi

# Compile deformable attention operators
echo ""
echo "========================================="
echo "Compiling Deformable Attention Operators"
echo "========================================="
echo ""

read -p "Compile deformable attention ops? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$MVGFORMER_DIR/lib/models/ops"

    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        echo "Detected CUDA version: $CUDA_VERSION"
        echo "CUDA_HOME: $CUDA_HOME"
    else
        echo "WARNING: nvcc not found. Please set CUDA_HOME manually."
        read -p "Enter CUDA_HOME path (e.g., /usr/local/cuda-11.8): " CUDA_HOME
    fi

    export CUDA_HOME
    python setup.py build install

    cd "$MVGFORMER_DIR"
    echo "✓ Deformable attention operators compiled"
else
    echo "⚠ Skipping deformable attention compilation"
fi

# Setup YOLO environment
echo ""
echo "========================================="
echo "Setting up YOLO environment"
echo "========================================="
echo ""

cd "$YOLO_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating UV virtual environment for YOLO..."
    uv venv --python 3.10
fi

source .venv/bin/activate
echo "Installing YOLO dependencies..."
uv sync

cd "$MVGFORMER_DIR"

# Create symbolic link to YOLO in MVGFormer (for easier imports)
if [ ! -L "yolo" ]; then
    echo ""
    echo "Creating symbolic link to YOLO..."
    ln -s "$YOLO_DIR/yolo" yolo
    echo "✓ Symbolic link created"
fi

# Download pretrained models (optional)
echo ""
echo "========================================="
echo "Download Pretrained Models (Optional)"
echo "========================================="
echo ""

mkdir -p "$MVGFORMER_DIR/models"

echo "Available models to download:"
echo "  1. PoseResNet50 (for baseline, 193 MB)"
echo "  2. YOLO v9-c wholebody (if available)"
echo "  3. Skip"

read -p "Select option (1-3): " -n 1 -r
echo

case $REPLY in
    1)
        echo "Downloading PoseResNet50..."
        wget -O "$MVGFORMER_DIR/models/pose_resnet50_panoptic.pth.tar" \
            "https://onedrive.live.com/download?cid=93774C670BD4F835&resid=93774C670BD4F835%211917&authkey=AMf08ZItxtILRuU"
        echo "✓ PoseResNet50 downloaded"
        ;;
    2)
        echo "YOLO wholebody weights should be placed at:"
        echo "  $YOLO_DIR/runs/train/v9-c/lightning_logs/version_X/checkpoints/best_c_XXXX_X.XXXX.pt"
        echo "Please train YOLO on wholebody34 dataset or specify the path in config"
        ;;
    *)
        echo "Skipping model download"
        ;;
esac

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate MVGFormer environment:"
echo "   cd $MVGFORMER_DIR"
echo "   source .venv/bin/activate"
echo ""
echo "2. Train with YOLO backbone:"
echo "   python -m torch.distributed.launch --nproc_per_node=8 --use_env \\"
echo "     run/train_3d.py --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml"
echo ""
echo "3. Or validate with YOLO backbone:"
echo "   python run/validate_3d.py \\"
echo "     --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml \\"
echo "     --model_path models/trained_model.pth.tar"
echo ""
echo "For more information, see docs/yolo_integration.md"
echo ""
