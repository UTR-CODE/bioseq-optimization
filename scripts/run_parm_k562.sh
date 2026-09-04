#!/usr/bin/env bash
# Launch the PARM K562 five-fold reproduction independently of a terminal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/hexin/miniconda3/envs/UTR/bin/python}"
RUN_NAME="strong_parm_k562_paper_v3_5fold_ensemble"
LOG_DIR="BioSeqBench/logs/${RUN_NAME}"
PID_FILE="${LOG_DIR}/runner.pid"
LOG_FILE="${LOG_DIR}/runner.stdout.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(<"$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "PARM training is already running (PID ${OLD_PID})."
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
fi

# `setsid` creates a new session, while `nohup` ignores terminal hangups.
# All additional arguments are forwarded to the Python trainer.
nohup setsid "$PYTHON_BIN" -u -m BioSeqBench.training.train_parm_k562_paper \
    --device cuda "$@" \
    >"$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Started PARM K562 training: PID ${PID}"
echo "Log: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop: kill ${PID}"
