# Prompt for Claude Code on the PC (inside WSL2)

## Prerequisite (one-time, do this first — Claude Code can't do it from inside WSL2)

In an **elevated PowerShell** on Windows:

```powershell
cd "C:\Users\ADMIN\Documents\Claude\Projects\AttackDRO"   # rename the folder first if still "First Paper"
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\setup_pc.ps1        # installs WSL2 + Ubuntu, checks the NVIDIA driver
```

Reboot if asked. Then open **Ubuntu** from the Start menu, `cd` into the repo
(e.g. `cd /mnt/c/Users/ADMIN/Documents/Claude/Projects/AttackDRO`, or clone it to
`~/projects/attackdro`), start **Claude Code**, and paste the prompt below.

---

## Paste this into Claude Code

```
You are running inside WSL2 (Ubuntu) on my Windows PC, in the AttackDRO repo. This PC has
an NVIDIA RTX 5070 Ti (Blackwell, sm_120). Set it up as a remote GPU training box that my
MacBook will drive over Tailscale. Work step by step, run the commands yourself, show me
the output of each, and STOP to ask whenever a step needs my browser or credentials.

There is a reference script at scripts/setup_wsl.sh — you may use it, but go step by step
and verify each stage rather than running it blind:

1. Confirm the GPU passes through to WSL2: run `nvidia-smi` and show me the RTX 5070 Ti.
   If it fails, stop and tell me to update the Windows NVIDIA driver + reboot (do NOT try
   to install a Linux driver inside WSL2).

2. Create a Python venv and install deps:
       python3 -m venv .venv && source .venv/bin/activate
       pip install --upgrade pip && pip install -r requirements.txt
   Then install PyTorch with CUDA 12.8 (Blackwell support):
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   Verify with `python scripts/check_gpu.py`. It must print CUDA available: True,
   compute capability (12, 0), and "Matmul on GPU: OK". If you see an sm_120 error,
   reinstall from the nightly index (.../whl/nightly/cu128) and re-verify.

3. Prove the training pipeline runs end to end:
       python src/train.py --config configs/default.yaml
   Confirm loss falls and accuracy rises, and that `nvidia-smi` shows GPU utilization
   while it runs.

4. Set up the SSH server so my Mac can connect into WSL2:
       sudo apt install -y openssh-server
       set Port to 2222 and PasswordAuthentication yes in /etc/ssh/sshd_config
       restart ssh, and make it auto-start on new shells
   Show me `sudo service ssh status`.

5. Install Tailscale (`curl -fsSL https://tailscale.com/install.sh | sh`), then STOP and
   tell me to run `sudo tailscale up` and authenticate in the browser. After I confirm,
   show me `tailscale ip -4` and remind me to rename this machine to "attackdro-pc" in the
   Tailscale admin console.

6. Start the job-dispatch watcher in the background so my Mac can submit jobs:
       nohup python jobs/agent.py > jobs/agent.out 2>&1 &
   Confirm it's running and watching jobs/queue/.

7. Help me push this repo to a NEW PRIVATE GitHub repo named "attackdro": set up an SSH
   key if needed (show me the public key to add to GitHub), then init/commit/push. Stop
   and wait for me to add the key to GitHub before pushing.

At the end, give me: this PC's Tailscale IP, my WSL username, and a one-line summary of
what's left for me to do on the MacBook.
```
