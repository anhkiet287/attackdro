#!/usr/bin/env bash
# AttackDRO - WSL2 (Ubuntu) setup (Stage 2)
# Run INSIDE WSL2 Ubuntu, from the repo root:
#     bash scripts/setup_wsl.sh
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root (script lives in scripts/)
REPO="$(pwd)"
echo "==> AttackDRO WSL2 setup  (repo: $REPO)"

# --- 1. System packages ---------------------------------------------------
echo "[1/6] Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git openssh-server curl

# --- 2. GPU passthrough check --------------------------------------------
echo "[2/6] Checking GPU passthrough..."
if ! nvidia-smi >/dev/null 2>&1; then
  echo "  [!] nvidia-smi failed inside WSL2."
  echo "      Update the Windows NVIDIA driver, reboot Windows, then re-run."
  echo "      (Do NOT install a Linux NVIDIA driver inside WSL2.)"
  exit 1
fi
nvidia-smi

# --- 3. Python venv + deps -----------------------------------------------
echo "[3/6] Creating virtualenv + installing deps..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- 4. PyTorch (cu128 for Blackwell / sm_120) ---------------------------
echo "[4/6] Installing PyTorch (CUDA 12.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "    Verifying GPU is visible to PyTorch..."
if ! python scripts/check_gpu.py | tee /tmp/_gpu.txt | grep -q "Matmul on GPU: OK"; then
  echo "  [!] Stable cu128 wheel did not pass. Trying nightly cu128..."
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  python scripts/check_gpu.py
fi

# --- 5. SSH server on port 2222 ------------------------------------------
echo "[5/6] Configuring SSH server (port 2222)..."
sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo service ssh restart 2>/dev/null || sudo service ssh start
# Auto-start ssh on new shells (WSL has no systemd by default)
grep -q "service ssh start" ~/.bashrc || \
  echo 'sudo service ssh start >/dev/null 2>&1' >> ~/.bashrc

# --- 6. Tailscale ---------------------------------------------------------
echo "[6/6] Installing Tailscale..."
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

echo ""
echo "================ ALMOST DONE — 2 manual steps ================"
echo "1) Authenticate Tailscale (opens a browser link):"
echo "       sudo tailscale up"
echo "   Then in the Tailscale admin console, rename this machine to: attackdro-pc"
echo ""
echo "2) Note this machine's Tailscale IP for the Mac to connect to:"
echo "       tailscale ip -4"
echo "============================================================="
echo "Then test:  python scripts/check_gpu.py  &&  python src/train.py --config configs/default.yaml"
