#!/usr/bin/env bash
# Sync the AttackDRO job queue between this machine (Mac) and the PC over Tailscale.
#   bash jobs/sync.sh push     # send queued jobs to the PC
#   bash jobs/sync.sh pull     # fetch logs + results from the PC
#   bash jobs/sync.sh          # push then pull
#
# Requires the `attackdro` SSH host alias in ~/.ssh/config (set up in MACBOOK_PROMPT.md).
set -euo pipefail

REMOTE="${ATTACKDRO_REMOTE:-attackdro}"                  # ssh host alias
REMOTE_REPO="${ATTACKDRO_REMOTE_REPO:-projects/attackdro}"  # path on the PC (relative to ~)
LOCAL_JOBS="$(cd "$(dirname "$0")" && pwd)"

push() {
  echo "==> push queued jobs -> $REMOTE"
  ssh "$REMOTE" "mkdir -p $REMOTE_REPO/jobs/queue"
  rsync -az -e ssh "$LOCAL_JOBS/queue/" "$REMOTE:$REMOTE_REPO/jobs/queue/"
}

pull() {
  echo "==> pull logs/done/failed <- $REMOTE"
  for d in logs done failed; do
    mkdir -p "$LOCAL_JOBS/$d"
    rsync -az -e ssh "$REMOTE:$REMOTE_REPO/jobs/$d/" "$LOCAL_JOBS/$d/" 2>/dev/null || true
  done
}

case "${1:-both}" in
  push) push ;;
  pull) pull ;;
  both) push; pull ;;
  *) echo "usage: $0 [push|pull]"; exit 1 ;;
esac
echo "done."
