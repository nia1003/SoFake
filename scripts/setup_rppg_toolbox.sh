#!/usr/bin/env bash
# scripts/setup_rppg_toolbox.sh
#
# One-shot setup: clone rPPG-Toolbox and download the PhysNet checkpoint.
# Run from the SoFake root directory:
#
#   bash scripts/setup_rppg_toolbox.sh
#
# Optional: choose a different model
#   MODEL=EfficientPhys DATASET=UBFC-rPPG bash scripts/setup_rppg_toolbox.sh

set -e

MODEL="${MODEL:-PhysNet}"
DATASET="${DATASET:-PURE}"
TOOLBOX_DIR="${RPPG_TOOLBOX_ROOT:-$(dirname "$0")/../rPPG-Toolbox}"

echo "=== rPPG-Toolbox setup ==="
echo "Toolbox dir : $TOOLBOX_DIR"
echo "Model       : $MODEL"
echo "Dataset     : $DATASET"
echo ""

# ── 1. Clone rPPG-Toolbox ───────────────────────────────────────────────
if [ ! -d "$TOOLBOX_DIR" ]; then
    echo "[1/3] Cloning nia1003/rPPG-Toolbox ..."
    git clone https://github.com/nia1003/rPPG-Toolbox.git "$TOOLBOX_DIR"
else
    echo "[1/3] rPPG-Toolbox already present at $TOOLBOX_DIR — skipping clone."
fi

# ── 2. Install Toolbox dependencies ─────────────────────────────────────
echo "[2/3] Installing rPPG-Toolbox requirements ..."
pip install -q -r "$TOOLBOX_DIR/requirements.txt" || true

# ── 3. Locate or download checkpoint ────────────────────────────────────
CKPT_DIR="$TOOLBOX_DIR/final_model_release"
CKPT_FILE="${DATASET}_${MODEL}_DiffNormalized.pth"
CKPT_PATH="$CKPT_DIR/$CKPT_FILE"

if [ -f "$CKPT_PATH" ]; then
    echo "[3/3] Checkpoint already exists: $CKPT_PATH"
else
    echo "[3/3] Checkpoint NOT found at $CKPT_PATH"
    echo ""
    echo "  rPPG-Toolbox hosts checkpoints on Google Drive."
    echo "  Please download manually from the repo README and place here:"
    echo "  $CKPT_PATH"
    echo ""
    echo "  Quick alternatives already in the repo (if present):"
    ls "$CKPT_DIR"/*.pth 2>/dev/null | head -10 || echo "  (none found yet)"
fi

# ── 4. Print usage ───────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Run single video:"
echo "  python3 evaluation/fusion_evaluator.py \\\\"
echo "      --video  path/to/video.mp4 \\\\"
echo "      --weights $CKPT_PATH \\\\"
echo "      --model  $MODEL"
echo ""
echo "Run full batch:"
echo "  python3 evaluation/batch_evaluate.py \\\\"
echo "      --output results/batch_toolbox"
echo ""
echo "Available models: PhysNet | EfficientPhys | PhysMamba | RhythmFormer |"
echo "                  FactorizePhys | PhysFormer | TS-CAN | DeepPhys | iBVPNet"
