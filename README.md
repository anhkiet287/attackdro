# AttackDRO

Research code for the **AttackDRO** project. Training runs on a Windows PC with an **RTX 5070 Ti**
(Blackwell, `sm_120`) inside **WSL2 Ubuntu**, accessed remotely from a **MacBook Pro M4**
over **Tailscale + SSH**.

## Architecture

```
MacBook Pro M4  ──Tailscale (encrypted, from anywhere)──►  Windows PC
   │                                                            │
   └─ VS Code Remote-SSH ──────────────► sshd inside WSL2 (Ubuntu)
                                              │
                                              └─ PyTorch + CUDA 12.8 → RTX 5070 Ti
```

The Mac is a thin client: you edit and launch runs from it, but all training executes
on the PC's GPU.

## First-time setup

See **[SETUP.md](SETUP.md)** for the full step-by-step (Tailscale, WSL2, NVIDIA driver,
PyTorch, VS Code Remote-SSH).

## Quick start (after setup)

```bash
# From the Mac: VS Code → Remote-SSH → connect to the PC, OR plain ssh:
ssh kiet@attackdro-pc        # Tailscale hostname

# On the PC (inside WSL2), one time:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install PyTorch separately — see SETUP.md (cu128 build)

# Sanity check the GPU is visible to PyTorch:
python scripts/check_gpu.py

# Train:
python src/train.py --config configs/default.yaml
```

## Layout

```
AttackDRO/
├── README.md
├── SETUP.md            # remote-GPU setup guide
├── requirements.txt    # deps (PyTorch installed separately, see SETUP.md)
├── .gitignore
├── configs/
│   └── default.yaml    # training hyperparameters
├── src/
│   └── train.py        # minimal PyTorch training loop (proves the GPU works)
├── scripts/
│   ├── check_gpu.py    # CUDA / device sanity check
│   ├── setup_pc.ps1    # PC setup stage 1 (WSL2 + Ubuntu, driver check)
│   └── setup_wsl.sh    # PC setup stage 2 (env, PyTorch, SSH, Tailscale)
├── jobs/               # Mac→PC job-dispatch queue (see jobs/README.md)
│   ├── agent.py        # watcher: runs on the PC, executes queued jobs on the GPU
│   ├── submit.py       # submit a job (from the Mac)
│   └── sync.sh         # rsync the queue/results over Tailscale
├── MACBOOK_PROMPT.md   # paste into Claude on the Mac to set up the client side
├── data/               # datasets (git-ignored)
└── notebooks/          # exploration (git-ignored outputs)
```

## Remote job dispatch (Mac → PC GPU)

Once set up, submit training runs from the Mac and let the PC's GPU execute them:

```bash
python jobs/submit.py --name exp1 -- python src/train.py --config configs/default.yaml
bash jobs/sync.sh              # push the job to the PC, pull back results
cat jobs/logs/<id>.log
```

See **[jobs/README.md](jobs/README.md)**. The PC runs `python jobs/agent.py` in the
background to watch the queue.
