# Setup: remote GPU access (MacBook M4 → Windows PC RTX 5070 Ti)

This walks through wiring your MacBook Pro M4 to train models on the Windows PC's
RTX 5070 Ti. You do each numbered step **once**. After that, a work session is just:
open VS Code on the Mac → connect → run training.

Conventions below: replace `kiet` with your WSL username and pick a Tailscale hostname
(this guide uses `attackdro-pc`).

---

## 0. Why this stack

- **Tailscale** — a private encrypted network between your two machines. No port
  forwarding, no exposing SSH to the public internet, works from campus/café.
- **WSL2 (Ubuntu)** — a real Linux running on Windows. This is what research labs use,
  and Blackwell GPUs (sm_120) are **not supported on native-Windows PyTorch** — WSL2 is
  the recommended path. Your scripts stay portable to Colab and a lab cluster.
- **SSH + VS Code Remote-SSH** — edit and run on the PC as if it were local.

---

## 1. Windows PC — NVIDIA driver

1. Install the latest **NVIDIA Game Ready or Studio driver** for the RTX 5070 Ti from
   nvidia.com (or GeForce Experience / NVIDIA App). The Windows driver is what exposes
   the GPU to WSL2 — **you do NOT install a separate CUDA driver inside WSL2.**
2. Reboot. Open PowerShell and confirm:
   ```powershell
   nvidia-smi
   ```
   You should see the RTX 5070 Ti and a CUDA version (12.8 or higher).

---

## 2. Windows PC — install WSL2 + Ubuntu

In an **Administrator PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted. Launch "Ubuntu" from the Start menu, create your Linux username
(`kiet`) and password. Then update and confirm the GPU passes through:

```bash
sudo apt update && sudo apt upgrade -y
nvidia-smi          # should show the RTX 5070 Ti from inside WSL2
```

If `nvidia-smi` works here, the hard part is done — CUDA passthrough is live.

> Keep your project on the **Linux filesystem** (e.g. `~/projects/attackdro`), not
> under `/mnt/c/...`. Linux-side files are much faster for training I/O.

---

## 3. WSL2 — Python env + PyTorch (cu128)

```bash
# Tools
sudo apt install -y python3 python3-venv python3-pip git

# Get the repo onto the PC (see section 7 for creating the GitHub remote first)
mkdir -p ~/projects && cd ~/projects
git clone git@github.com:<your-username>/attackdro.git
cd attackdro

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Project deps (PyTorch is NOT in requirements.txt — installed next)
pip install -r requirements.txt
```

**Install PyTorch with CUDA 12.8** (Blackwell/sm_120 support). Use the official selector
at https://pytorch.org/get-started/locally/ and pick: Linux / Pip / Python / CUDA 12.8.
As of now that command is:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If the stable cu128 wheel doesn't yet detect sm_120 on your setup, fall back to nightly:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Verify the GPU is usable from PyTorch:**

```bash
python scripts/check_gpu.py
```

Expected: `CUDA available: True`, device `RTX 5070 Ti`, compute capability `(12, 0)`,
and `Matmul on GPU: OK`. If you see a `sm_120 not supported` warning, switch to the
nightly wheel above.

Then run the toy training to prove the whole loop end to end:

```bash
python src/train.py --config configs/default.yaml
```

---

## 4. Both machines — Tailscale

1. Make a free account at https://tailscale.com (sign in with your Google account).
2. **Windows PC:** install Tailscale for Windows, sign in.
   - Also install it **inside WSL2** so the Linux side has its own address:
     ```bash
     curl -fsSL https://tailscale.com/install.sh | sh
     sudo tailscale up
     ```
     Follow the printed URL to authenticate.
3. **MacBook:** install Tailscale from the Mac App Store (or `brew install tailscale`),
   sign in with the **same account**.
4. In the Tailscale admin console (https://login.tailscale.com/admin/machines), find the
   PC and rename it to `attackdro-pc` so you get a stable hostname. (Enable MagicDNS if
   prompted — it lets you use the name instead of an IP.)

Test from the Mac terminal:
```bash
tailscale ping attackdro-pc
```

---

## 5. WSL2 — SSH server

WSL2 needs its own SSH daemon so the Mac can connect into Linux directly.

```bash
sudo apt install -y openssh-server

# Use a non-default port to avoid clashing with Windows' own SSH (which uses 22).
sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config

sudo service ssh restart
sudo service ssh status      # confirm it's running
```

> **WSL networking note:** by default WSL2 has its own internal IP, so reaching it from
> the Mac is easiest when Tailscale runs *inside* WSL2 (step 4.2) — then the Mac connects
> straight to the WSL Tailscale address. Run `tailscale ip -4` inside WSL2 to get it, or
> use the WSL machine's name from the admin console.
>
> Make SSH auto-start: add `sudo service ssh start` to your `~/.bashrc`, or enable
> systemd in `/etc/wsl.conf` (`[boot]\nsystemd=true`) and `sudo systemctl enable ssh`.

---

## 6. MacBook — SSH keys + connect

```bash
# On the Mac, generate a key if you don't have one:
ssh-keygen -t ed25519 -C "kiet@macbook"

# Copy it to the PC (uses the WSL Tailscale hostname/IP from step 5):
ssh-copy-id -p 2222 kiet@<wsl-tailscale-name>

# Connect:
ssh -p 2222 kiet@<wsl-tailscale-name>
```

Add a shortcut to `~/.ssh/config` on the Mac so you can just type `ssh attackdro`:

```
Host attackdro
    HostName <wsl-tailscale-name>
    User kiet
    Port 2222
```

### VS Code Remote-SSH (the main workflow)

1. On the Mac, install **VS Code** and the **Remote - SSH** extension.
2. Cmd+Shift+P → "Remote-SSH: Connect to Host" → `attackdro`.
3. VS Code opens with the PC's filesystem. Open `~/projects/attackdro`, pick the
   `.venv` interpreter, and use the integrated terminal to run training. Code lives on
   the PC; the GPU does the work; you drive it all from the Mac.

> Notebooks: VS Code's Jupyter support runs kernels remotely over this same SSH session,
> so `.ipynb` files "just work" on the GPU too.

---

## 7. GitHub — private repo + push (do this first to get the scaffold online)

The scaffold currently lives in your Windows folder
`C:\Users\ADMIN\Documents\Claude\Projects\AttackDRO`. Put it on GitHub, then clone it
into WSL2 (section 3).

**7a. Rename the folder and delete the stray `.git`.** The folder on disk is still named
`First Paper`, and a half-initialized, corrupted `.git` was left inside it. Fix both in
**PowerShell**:

```powershell
cd "C:\Users\ADMIN\Documents\Claude\Projects"
Rename-Item "First Paper" "AttackDRO"
cd "AttackDRO"
Remove-Item -Recurse -Force .git
```

**7b. Create a private repo** named `attackdro` on github.com (empty — no README, no
.gitignore, since we already have them).

**7c. Init, commit, push.** Easiest to run inside **WSL2** so your keys live in Linux.
Either work from the Windows path (`cd "/mnt/c/Users/ADMIN/Documents/Claude/Projects/AttackDRO"`)
or just copy the files into `~/projects/attackdro`. Then:

```bash
git init -b main
git add -A
git commit -m "Initial scaffold: PyTorch project + remote-GPU setup guide"
git remote add origin git@github.com:<your-username>/attackdro.git
git push -u origin main
```

Set up a GitHub SSH key on the PC if you haven't:
```bash
ssh-keygen -t ed25519 -C "kiet-pc"
cat ~/.ssh/id_ed25519.pub     # add this to GitHub → Settings → SSH keys
```

> After it's on GitHub, the canonical copy you train from is the WSL2 clone in
> `~/projects/attackdro` (section 3) — fast Linux-filesystem I/O.

---

## 8. Verify everything works (test checklist)

Run these in order. Each one confirms a layer; if one fails, fix it before moving on.

**PC side (run inside WSL2 on the Windows PC):**

| # | Test | Command | Pass looks like |
|---|------|---------|-----------------|
| 1 | GPU passthrough | `nvidia-smi` | Table showing **RTX 5070 Ti** |
| 2 | PyTorch sees GPU | `python scripts/check_gpu.py` | `CUDA available: True`, capability `(12, 0)`, `Matmul on GPU: OK` |
| 3 | Training loop runs | `python src/train.py --config configs/default.yaml` | Loss falls, accuracy rises over 5 epochs, `done.` |
| 4 | GPU is actually used | In a 2nd terminal during test 3: `watch -n0.5 nvidia-smi` | GPU-Util jumps above 0%, python process listed |

**Network + Mac side:**

| # | Test | Where | Command | Pass looks like |
|---|------|-------|---------|-----------------|
| 5 | Tailscale link | Mac | `tailscale ping attackdro-pc` | `pong from attackdro-pc ...` |
| 6 | SSH in | Mac | `ssh attackdro` | You land in the PC's WSL2 shell |
| 7 | Remote GPU | Mac (over SSH) | `cd ~/projects/attackdro && source .venv/bin/activate && python scripts/check_gpu.py` | Same as test 2 — GPU reported from the remote shell |
| 8 | End-to-end | Mac → VS Code Remote-SSH | Run `python src/train.py` in VS Code's terminal; watch `nvidia-smi` | Training runs on the PC's GPU, driven from the Mac |

If tests 1–4 pass you have a working GPU box; if 5–8 pass you have full remote access from
the MacBook. Test 8 is the real goal: editing and launching from the Mac, computing on the PC.

---

## Daily workflow (after all of the above)

1. PC is on, Tailscale running (it auto-starts).
2. Mac: open VS Code → Remote-SSH → `attackdro`.
3. Terminal: `source .venv/bin/activate && python src/train.py --config configs/default.yaml`.
4. Commit and push from the same terminal.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi` fails in WSL2 | Update the Windows NVIDIA driver; reboot. Don't install a Linux driver inside WSL2. |
| `CUDA available: False` in PyTorch | Wrong wheel — reinstall with `--index-url .../cu128`. |
| `sm_120 is not supported` | Use the nightly cu128 wheel (step 3). |
| Can't SSH from Mac | Confirm `tailscale ping attackdro-pc` works; confirm `sudo service ssh status` in WSL2; check the port (2222). |
| SSH dies when WSL closes | WSL shuts down when the last shell exits — keep a terminal open, or enable systemd + `wsl --shutdown` discipline. Keeping the VS Code session open holds it up. |
| Training slow on I/O | Move data onto the Linux filesystem (`~/`), not `/mnt/c`. |

---

### Sources

- [PyTorch sm_120 / Blackwell support (issue #164342)](https://github.com/pytorch/pytorch/issues/164342)
- [WSL2 Blackwell PyTorch setup guide](https://medium.com/@getnetdemil/getting-pytorch-to-actually-use-your-rtx-5090-a-complete-wsl2-setup-guide-for-blackwell-sm-120-61f86f64abc4)
- [Fix PyTorch sm_120 on RTX Blackwell — cu128 setup](https://medium.com/@harishpillai1994/fix-pytorch-sm-120-on-rtx-blackwell-gpus-cuda-docker-cu128-setup-to-run-llms-44f25179ac76)
- [PyTorch install selector](https://pytorch.org/get-started/locally/)
