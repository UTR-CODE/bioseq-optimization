#!/usr/bin/env bash
# Launch/resume BioSeqBench's 10-fold MPRA-LegNet suite outside the terminal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/hexin/miniconda3/envs/UTR/bin/python}"
RUN_NAME="strong_agarwal2025_mpralegnet_v1"
LOG_DIR="BioSeqBench/logs/${RUN_NAME}"
PID_FILE="${LOG_DIR}/runner.pid"
LOG_FILE="${LOG_DIR}/runner.stdout.log"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(<"$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "MPRALegNet training is already running (PID ${OLD_PID})."
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
fi

# `setsid` creates a new session and `nohup` ignores terminal hangups.  The
# trainer skips already completed fold pairs, including the two valid HepG2
# models produced before the interrupted exhaustive-CV attempt.
nohup setsid "$PYTHON_BIN" -u -m BioSeqBench.training.train_agarwal_mpralegnet \
    --mode 10fold --skip-complete --device cuda \
    >"$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Started/resumed MPRALegNet 10-fold suite: PID ${PID}"
echo "Log: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop: kill ${PID}"
