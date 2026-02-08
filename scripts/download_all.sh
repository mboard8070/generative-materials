#!/bin/bash
#
# Download all training data from Poly Haven and AmbientCG
#
# Estimated downloads:
#   Poly Haven: ~600 materials, ~3-5GB
#   AmbientCG:  ~1000 materials, ~5-8GB
#   Total:      ~1600 materials, ~8-13GB
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo "  Text-to-Material Training Data Downloader"
echo "============================================"
echo ""

# Check disk space
AVAILABLE=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
echo "Available disk space: ${AVAILABLE}GB"

if [ "$AVAILABLE" -lt 20 ]; then
    echo "WARNING: Less than 20GB available. Downloads may fail."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1/3: Downloading from Poly Haven..."
echo "----------------------------------------"
python3 scripts/download_polyhaven.py

echo ""
echo "Step 2/3: Downloading from AmbientCG..."
echo "---------------------------------------"
python3 scripts/download_ambientcg.py

echo ""
echo "Step 3/3: Preparing training dataset..."
echo "---------------------------------------"
python3 scripts/prepare_dataset.py

echo ""
echo "============================================"
echo "  Download Complete!"
echo "============================================"
echo ""
echo "Raw data:     data/raw/"
echo "Training set: data/training/"
echo ""
echo "Next step: Run LoRA training with:"
echo "  python3 scripts/train_lora.py"
