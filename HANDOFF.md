# AttackDRO — Remote GPU Training Box: Handoff

A one-page reference for what was built, how to use it, and how to recover it.
Read this first if you're picking up the project after time away, or if you want
to design a prototype (UI, dashboard, job submitter, etc.) on top of this stack.

---

## 1. What this is

A self-hosted remote GPU training box. The **Windows PC** (RTX 5070 Ti) runs
Ubuntu inside WSL2 and exposes:
- An **SSH server** for interactive work and code-sync from a **MacBook** client.
- A **job-queue agent** that watches a directory and runs queued commands on the GPU.
- **Tailscale** as the secure network layer — no public IP, no port forwarding,
  works on any network the laptop happens to be on.

The MacBook is the driver; the PC is the executor.

---

## 2. Architecture

```
+----------------+           Tailscale (WireGuard mesh)           +-------------------+
|  MacBook M4    |  <---------------------------------------->   |  Windows PC       |
|  nguyens-      |       100.78.145.71 <----> 100.70.237.42      |  RTX 5070 Ti      |
|  macbook-pro   |                                                |  (kiet-pc)        |
+----------------+                                                +---------+---------+
        |                                                                   |
        | ssh -p 2222                                                       v
        | scp / rsync                                          +------------------------+
        | VS Code Remote-SSH                                   | WSL2 Ubuntu 24.04      |
        | drop JSON into jobs/queue/                           |  - sshd on :2222       |
        v                                                      |  - tailscaled          |
                                                               |  - attackdro-agent     |
                                                               |    (watches queue)     |
                                                               |  - .venv: PyTorch      |
                                                               |    2.11.0 + cu128      |
                                                               +------------------------+
                                                                          |
                                                                          v
                                                               GPU: sm_120 (Blackwell)
```

---

## 3. Specifications (the actual values)

| Item | Value |
|---|---|
| PC Windows user | `ADMIN` |
| WSL distro | `Ubuntu` (24.04.4) |
| WSL Linux user | `endmin` |
| Repo path (WSL view) | `/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO` |
| Repo path (Windows view) | `C:\Users\ADMIN\Documents\Claude\Projects\ATTACKDRO` |
| Repo remote (GitHub, private) | `git@github.com:anhkiet287/attackdro.git` |
| Default branch | `main` |
| PC Tailscale IPv4 | `100.70.237.42` |
| PC Tailscale name | `kiet-pc` (rename to `attackdro-pc` in admin console — optional) |
| Mac Tailscale IPv4 | `100.78.145.71` |
| SSH port (inside WSL) | `2222` |
| Python env | `/mnt/c/.../ATTACKDRO/.venv` — `python3.12`, `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` |
| GPU compute capability | `(12, 0)` Blackwell, 16 GB VRAM |
| Driver / CUDA (host) | NVIDIA driver `591.86`, CUDA runtime `13.1` |

---

## 4. Auto-start chain (no manual steps after PC boots)

On Windows user **login**:
1. `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AttackDRO-AutoStart.vbs`
   silently runs `wsl.exe -d Ubuntu --exec /bin/bash -lc "exec tail -f /dev/null"`.
   This boots the WSL2 distro and pins it alive (no console window).
2. Inside WSL2, systemd starts three services automatically:

| Service | What it does | Where defined |
|---|---|---|
| `ssh.socket` | Listens on `0.0.0.0:2222` + `[::]:2222` | `/etc/systemd/system/ssh.socket.d/override.conf` |
| `tailscaled.service` | Joins the tailnet, owns `100.70.237.42` | (default install) |
| `attackdro-agent.service` | Runs `jobs/agent.py`, polls `jobs/queue/` every 3s, auto-restarts on crash | `/etc/systemd/system/attackdro-agent.service` |

Manual fallback for non-technical users: double-click **`Desktop\Start AttackDRO.lnk`**.

---

## 5. Daily workflow

**From the MacBook:**
```bash
ssh attackdro                                # opens WSL shell on the PC
# or
code --remote ssh-remote+attackdro /mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO
```

The Mac's `~/.ssh/config` has:
```
Host attackdro
    HostName 100.70.237.42
    User endmin
    Port 2222
```

**Two ways to run a training job:**

1. **Interactive** — SSH in, activate venv, run `python src/train.py --config configs/default.yaml`.
2. **Queued (fire-and-forget)** — drop a JSON spec into `jobs/queue/`:
   ```json
   {"id":"exp042","cmd":["python","src/train.py","--config","configs/default.yaml","--epochs","50"]}
   ```
   The agent picks it up within 3s, moves it to `jobs/running/`, writes logs to
   `jobs/logs/<id>.log`, then moves it to `jobs/done/` or `jobs/failed/`.

---

## 6. Job-queue protocol (for whoever designs a prototype UI)

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Unique string. Used as filename stem and log filename. |
| `cmd` | yes | List of strings, passed to `subprocess.run` (no shell). |
| `cwd` | no | Path relative to repo root. Defaults to repo root. |
| `status` | (set by agent) | `running` → `done` \| `failed` |
| `started_at`, `finished_at`, `exit_code` | (set by agent) | ISO 8601 timestamps. |

Directory layout (gitignored except for `.gitkeep` markers):
```
jobs/
  queue/    # submit here
  running/  # currently executing
  done/     # exit 0
  failed/   # nonzero exit or agent error
  logs/     # per-job stdout+stderr
  agent.out # (legacy nohup output, unused with systemd)
```

Agent source: `jobs/agent.py` (~90 lines, no deps beyond stdlib).

---

## 7. Recovery cheatsheet

| Symptom | Fix |
|---|---|
| Mac can't SSH to `attackdro` | On PC: `wsl -d Ubuntu -- systemctl status ssh.socket attackdro-agent`. If WSL itself is down, double-click `Start AttackDRO` desktop shortcut. |
| Agent stuck / not picking up jobs | `sudo systemctl restart attackdro-agent.service` inside WSL. |
| Tailscale offline | `sudo systemctl restart tailscaled` inside WSL. If still bad: `sudo tailscale up` and re-auth in browser. |
| PyTorch broken (e.g. driver update) | `source .venv/bin/activate && python scripts/check_gpu.py`. If sm_120 error: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`. |
| Need WSL/PC logs | `journalctl -u attackdro-agent -n 200` for agent, `journalctl -u ssh.socket -n 50` for SSH. |
| Repo on PC is broken | It's a clone of `git@github.com:anhkiet287/attackdro.git`. Worst case: `git clone` fresh to a new path, recreate `.venv`. |

---

## 8. Limitations / known sharp edges

- **Repo lives on `/mnt/c/`** (NTFS via DrvFs). Pip/git ops are 2-5× slower than
  native ext4; bulk data I/O during training is slower than ideal. Tolerable for
  small/medium models. Migrate to `~/projects/attackdro` on ext4 if it becomes
  a bottleneck.
- **No GPU sharing/queueing.** The agent runs **one job at a time** in FIFO order.
  Long jobs block the queue.
- **Single Windows user.** Auto-start only fires for the `ADMIN` Windows user.
  WSL distro shuts down when that user logs out.
- **Sudo elevation needed for some maintenance** — only `service ssh start|status|restart`
  is passwordless (set in `/etc/sudoers.d/attackdro-ssh`). All other admin tasks
  prompt for the WSL password.
- **`kiet-pc` not renamed in Tailscale admin console.** SSH config uses the raw IP
  so this doesn't matter functionally, but renaming to `attackdro-pc` would let
  you switch `HostName 100.70.237.42` → `HostName attackdro-pc` for clarity.

---

## 9. Where things came from (so you can find them later)

| Created/edited file | Purpose |
|---|---|
| `/etc/systemd/system/attackdro-agent.service` | Unit file for the job agent |
| `/etc/systemd/system/ssh.socket.d/override.conf` | Moves sshd to port 2222, IPv4+IPv6 |
| `/etc/sudoers.d/attackdro-ssh` | Passwordless `service ssh start/status/restart` |
| `~/.ssh/id_ed25519[.pub]` (WSL) | Key registered with GitHub for `anhkiet287` |
| `~/attackdro_*.sh` (WSL, several) | One-shot setup scripts kept for reference |
| `%APPDATA%\...\Startup\AttackDRO-AutoStart.vbs` | Hidden launcher fired at Windows login |
| `%USERPROFILE%\Desktop\Start AttackDRO.lnk` | Manual launcher for the parent |
| `scripts/setup_pc.ps1`, `scripts/setup_wsl.sh`, `scripts/check_gpu.py` | Bootstrap helpers in-repo |

---

## 10. If you're designing a prototype on top of this

The infrastructure above gives you, for free, the primitives any prototype needs:

- **Execute arbitrary commands on the GPU**: write a JSON to `jobs/queue/`, get a
  log file back. No HTTP server needed.
- **Stream logs**: tail `jobs/logs/<id>.log` over SSH (`ssh attackdro tail -f ...`).
- **Read results**: model checkpoints, plots, etc. are at conventional paths under
  the repo; pull with `scp` or `rsync` over the same SSH connection.
- **List job state**: `ls jobs/queue jobs/running jobs/done jobs/failed`.

Sensible prototypes to build on this:
1. A **web/desktop UI** that lets a non-technical user pick a config, click "Run",
   and watch live logs (just shells out to `ssh attackdro` under the hood).
2. A **Mac menu-bar app** showing PC online/offline + current job + GPU utilization
   (`ssh attackdro nvidia-smi --query-gpu=... --format=csv`).
3. A **multi-tenant queue extension** — replace the JSON-on-disk agent with a
   small SQLite-backed scheduler that respects job priorities and GPU memory.

Anything in the "Limitations" section above is fair game to redesign.
