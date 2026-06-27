# Job dispatch (Mac → PC GPU)

A file-based queue so the MacBook can hand training jobs to the PC's RTX 5070 Ti.
No AI involved — just a folder the PC watches.

```
Mac:  submit.py ─► jobs/queue/*.json ──(sync.sh push, rsync/Tailscale)──►  PC: jobs/queue/
                                                                              │
                                                            PC: agent.py watches queue/, runs
                                                            the job on the GPU, writes:
                                                              jobs/logs/<id>.log
                                                              jobs/done/<id>.json   (exit 0)
                                                              jobs/failed/<id>.json (exit != 0)
                                                                              │
Mac:  jobs/{logs,done,failed}/  ◄──(sync.sh pull, rsync/Tailscale)───────────┘
```

## On the PC (once): start the watcher

```bash
cd ~/projects/attackdro
source .venv/bin/activate
nohup python jobs/agent.py > jobs/agent.out 2>&1 &   # runs in the background
```

It claims jobs atomically (moves `queue/ → running/`) and runs them one at a time.

## On the Mac: submit and check results

```bash
# 1. Create a job (everything after -- is run on the PC, from the repo root)
python jobs/submit.py --name exp1 -- python src/train.py --config configs/default.yaml

# 2. Send it to the PC, then later pull results back
bash jobs/sync.sh push
bash jobs/sync.sh pull

# 3. Read the log
cat jobs/logs/<id>.log
```

`bash jobs/sync.sh` with no argument does push then pull.

## Notes

- One job at a time (single GPU). Submit several; they run in order.
- A job spec is just JSON: `{ id, name, cmd, cwd, status, exit_code, ... }`.
- Override the remote in the environment if your paths differ:
  `ATTACKDRO_REMOTE=attackdro ATTACKDRO_REMOTE_REPO=projects/attackdro bash jobs/sync.sh`
- Runtime contents of `jobs/` are git-ignored; the folder structure is kept via `.gitkeep`.
- To watch a running job live, SSH in: `ssh attackdro 'tail -f ~/projects/attackdro/jobs/logs/<id>.log'`
