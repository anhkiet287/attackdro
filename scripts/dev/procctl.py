"""procctl — strict process matching for this repo's long-running jobs. DRY-RUN BY DEFAULT.

Written after two self-match incidents: `pkill -f <pattern>` and `ps | grep <substring> | kill`
both matched things that merely *mentioned* the pattern — the invoking shell itself, and the
`bash -c` wrapper of an unrelated job whose command line happened to contain the script name.
Killing a running 12-AA @10k audit costs ~7 h, so matching must be exact, not substring-ish.

Rules enforced here:
  * argv[0] must be a python interpreter (never a shell, never a `bash -c` wrapper)
  * argv[1] must be the exact script path/basename requested
  * every additional --flag=value constraint must appear as an exact argv token pair
  * the caller's own PID and its whole ancestor chain are always excluded
  * nothing is signalled unless --kill is passed; the default prints what WOULD be hit

  python scripts/dev/procctl.py --script c5_fromscratch.py --arg --outdir=.../M1a_cleance
  python scripts/dev/procctl.py --script eval_multinorm_audit.py            # list audits
  python scripts/dev/procctl.py --script c5_fromscratch.py --arg ... --kill # act
"""
from __future__ import annotations
import argparse, os, signal, subprocess, sys


def ancestors(pid):
    """PID plus every parent up to init — never signal our own lineage."""
    out, cur = set(), pid
    while cur and cur > 1 and cur not in out:
        out.add(cur)
        try:
            cur = int(subprocess.run(["ps", "-o", "ppid=", "-p", str(cur)],
                                     capture_output=True, text=True).stdout.strip() or 0)
        except ValueError:
            break
    return out


def procs():
    """(pid, ppid, etime, argv[]) for every process, argv split on NUL from /proc when possible."""
    r = subprocess.run(["ps", "-eo", "pid=,ppid=,etime=,args="], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, args = parts
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a for a in f.read().decode("utf-8", "replace").split("\0") if a]
        except (OSError, ValueError):
            argv = args.split()
        if argv:
            yield int(pid), int(ppid), etime, argv


def is_python(a0):
    return os.path.basename(a0).startswith("python") or a0.endswith("/python") or a0.endswith("python3")


def match(script, argpairs, exclude):
    hits = []
    for pid, ppid, etime, argv in procs():
        if pid in exclude:
            continue
        if len(argv) < 2 or not is_python(argv[0]):
            continue                                   # excludes shells and `bash -c` wrappers
        if os.path.basename(argv[1]) != os.path.basename(script):
            continue
        ok = True
        for flag, val in argpairs:
            # exact token pair "--flag value", or exact "--flag=value"
            paired = any(argv[i] == flag and i + 1 < len(argv) and argv[i + 1] == val
                         for i in range(len(argv)))
            joined = f"{flag}={val}" in argv
            if not (paired or joined):
                ok = False
                break
        if ok:
            hits.append((pid, ppid, etime, argv))
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True, help="exact script basename, e.g. c5_fromscratch.py")
    p.add_argument("--arg", action="append", default=[],
                   help="repeatable --flag=value constraint, matched as an exact argv token pair")
    p.add_argument("--kill", action="store_true", help="actually signal (default: dry run)")
    p.add_argument("--signal", default="TERM", choices=["TERM", "KILL", "STOP", "CONT"])
    p.add_argument("--parents-only", action="store_true",
                   help="drop hits whose ppid is also a hit (dataloader workers share the parent's argv)")
    p.add_argument("--expect", type=int, default=None,
                   help="refuse to act unless exactly this many processes match")
    a = p.parse_args()

    pairs = []
    for s in a.arg:
        if "=" not in s:
            sys.exit(f"--arg must be --flag=value, got {s!r}")
        f, v = s.split("=", 1)
        pairs.append((f, v))

    excl = ancestors(os.getpid())
    hits = match(a.script, pairs, excl)
    if a.parents_only:
        pids = {h[0] for h in hits}
        hits = [h for h in hits if h[1] not in pids]

    print(f"[procctl] script={a.script} constraints={pairs or '(none)'}")
    print(f"[procctl] excluded self+ancestors: {sorted(excl)}")
    if not hits:
        print("[procctl] NO MATCH")
        return 0 if not a.kill else 1
    for pid, ppid, etime, argv in hits:
        print(f"  pid={pid:<8} ppid={ppid:<8} etime={etime:<12} {' '.join(argv)[:130]}")

    if a.expect is not None and len(hits) != a.expect:
        sys.exit(f"[procctl] REFUSING: expected {a.expect} match(es), found {len(hits)}")
    if not a.kill:
        print(f"[procctl] DRY RUN — would send SIG{a.signal} to {[h[0] for h in hits]}")
        return 0
    for pid, *_ in hits:
        os.kill(pid, getattr(signal, f"SIG{a.signal}"))
        print(f"[procctl] SIG{a.signal} -> {pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
