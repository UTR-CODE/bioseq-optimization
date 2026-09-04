#!/usr/bin/env bash
# Full source-compatible MPRALegNet reproduction for Agarwal et al. lentiMPRA.
#
# Runs 90 ordered validation/test combinations per cell type.  For each outer
# test fold it averages the nine models and then averages the two *physically
# assayed* orientations of each element before scoring Table S8.  It is safe to
# rerun: completed fold directories containing result.json are retained.
#
# Examples:
#   BIOSEQBENCH_CELLS=hepg2 BIOSEQBENCH_EVALUATE_ONLY=1 bash "$0"
#   BIOSEQBENCH_CELLS=hepg2,k562,wtc11 bash "$0"

set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$script_root"

biosbench_python="${BIOSEQBENCH_PYTHON:-/home/hexin/miniconda3/envs/UTR/bin/python}"
biosbench_checkpoint_dir="BioSeqBench/checkpoints/strong_agarwal2025_mpralegnet_rev0_avg_4ch_v2"
biosbench_log_dir="BioSeqBench/logs/strong_agarwal2025_mpralegnet_rev0_avg_4ch_v2"
biosbench_results_file="BioSeqBench/results/agarwal2025_mpralegnet_rev0_avg_4ch_v2.csv"
biosbench_summary_file="BioSeqBench/results/agarwal2025_mpralegnet_paper_orientation_cv.csv"
biosbench_average_dir="BioSeqBench/raw_data/agarwal2025_lentimpra/human_legnet"
biosbench_supplement_dir="BioSeqBench/raw_data/agarwal2025_lentimpra"
biosbench_cells="${BIOSEQBENCH_CELLS:-hepg2,k562,wtc11}"
biosbench_evaluate_only="${BIOSEQBENCH_EVALUATE_ONLY:-0}"

if [[ ! -x "$biosbench_python" ]]; then
  echo "Python interpreter not executable: $biosbench_python" >&2
  echo "Set BIOSEQBENCH_PYTHON to the Python executable for the BioSeqBench environment." >&2
  exit 2
fi

for biosbench_average_file in HepG2_averaged.tsv K562_averaged.tsv WTC11_averaged.tsv; do
  if [[ ! -s "$biosbench_average_dir/$biosbench_average_file" ]]; then
    echo "Missing author averaged evaluation table: $biosbench_average_dir/$biosbench_average_file" >&2
    exit 2
  fi
done

if [[ ! -s "$biosbench_supplement_dir/2023-03-03628-s5/2023-03-03628C-Table S8 - folds and performance.xlsx" ]]; then
  echo "Missing Supplementary Table S8 required for the paper-equivalent score." >&2
  exit 2
fi

# Do not race a still-running previous invocation that writes the same fold
# checkpoints.  The current WTC11 run can finish first; a later invocation will
# resume it using --skip-complete.
if [[ "$biosbench_evaluate_only" != "1" ]] && pgrep -f 'BioSeqBench.training.train_agarwal_mpralegnet.*strong_agarwal2025_mpralegnet_rev0_avg_4ch_v2' >/dev/null; then
  echo "An MPRALegNet paper-CV process is already running for this checkpoint directory." >&2
  echo "Wait for it to finish, then rerun this script; completed folds will be skipped." >&2
  exit 3
fi

mkdir -p "$biosbench_log_dir"

if [[ "$biosbench_evaluate_only" != "1" ]]; then
  IFS=',' read -r -a biosbench_cell_list <<< "$biosbench_cells"
  for biosbench_cell in "${biosbench_cell_list[@]}"; do
    biosbench_cell="${biosbench_cell//[[:space:]]/}"
    case "$biosbench_cell" in hepg2|k562|wtc11) ;; *) echo "Unknown cell: $biosbench_cell" >&2; exit 2;; esac
    echo "[$(date --iso-8601=seconds)] Starting/resuming $biosbench_cell full 10x9 CV"
    "$biosbench_python" -u -m BioSeqBench.training.train_agarwal_mpralegnet \
      --cells "$biosbench_cell" \
      --mode cv \
      --device cuda \
      --no-use-reverse-channel \
      --skip-complete \
      --checkpoint-dir "$biosbench_checkpoint_dir" \
      --log-dir "$biosbench_log_dir" \
      --results-file "$biosbench_results_file"
  done
fi

echo "[$(date --iso-8601=seconds)] Aggregating nine-model outer-fold ensembles with physical-orientation pairing"
"$biosbench_python" -u -m BioSeqBench.evaluation.aggregate_agarwal_mpralegnet_paper_cv \
  --checkpoint-dir "$biosbench_checkpoint_dir" \
  --raw-data-root "$biosbench_average_dir" \
  --supplement-root "$biosbench_supplement_dir" \
  --cells "$biosbench_cells" \
  --output "$biosbench_summary_file"

echo "Completed. Summary: $biosbench_summary_file"
