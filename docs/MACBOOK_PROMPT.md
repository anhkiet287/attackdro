# Prompt to paste into Claude (Cowork) on your MacBook

Copy everything in the box below into Claude on your MacBook Pro M4. It will set up
and test the Mac side of the AttackDRO remote-GPU workflow.

> **Before you paste:** on the PC you should already have run `setup_wsl.sh`, run
> `sudo tailscale up`, renamed the PC to `attackdro-pc` in the Tailscale admin console,
> and noted its Tailscale IP (`tailscale ip -4`). Have that IP ready.

---

```
You are helping me set up my MacBook Pro M4 as a remote client to train ML models on my
Windows PC's RTX 5070 Ti GPU. The PC runs Ubuntu under WSL2 and is already set up:
PyTorch (CUDA 12.8) works, an SSH server listens on PORT 2222 inside WSL2, and Tailscale
is running with the machine named "attackdro-pc". My WSL username is <FILL IN> and the
PC's Tailscale IP is <FILL IN>.

Please do the Mac-side setup and testing, step by step, confirming each step before
moving on:

1. Install Tailscale on this Mac (Homebrew or Mac App Store), have me sign in with the
   SAME account as the PC, then verify connectivity:
       tailscale ping attackdro-pc
   It should return "pong".

2. Generate an SSH key if I don't have one (ed25519), and copy it to the PC:
       ssh-copy-id -p 2222 <wsl-user>@attackdro-pc
   (Use the Tailscale IP if the hostname doesn't resolve.)

3. Add a convenience host to ~/.ssh/config so I can just type `ssh attackdro`:
       Host attackdro
           HostName attackdro-pc
           User <wsl-user>
           Port 2222
   Then test: `ssh attackdro` should drop me into the PC's WSL2 shell.

4. Install VS Code + the "Remote - SSH" extension. Walk me through connecting to the
   `attackdro` host in VS Code and opening ~/projects/attackdro.

5. Run the remote test checklist and report results:
   a. Over SSH: `cd ~/projects/attackdro && source .venv/bin/activate && python scripts/check_gpu.py`
      -> expect CUDA available: True, compute capability (12, 0), Matmul on GPU: OK
   b. End-to-end: in VS Code's remote terminal run
      `python src/train.py --config configs/default.yaml`
      and confirm it trains on the PC's GPU (loss falls, accuracy rises).

Explain anything I need to click or authenticate, and stop and ask me if a step fails.
```

---

Replace `<FILL IN>` / `<wsl-user>` with your real WSL username and the PC's Tailscale IP
before sending.
