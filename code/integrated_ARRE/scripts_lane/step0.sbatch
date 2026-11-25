#!/usr/bin/env bash
# Lane A - Step-0 diagnostic (one GPU, read-only w.r.t. the shared tree; writes only to
# $CODE/docs/lane_a and $TMPDIR).
#   usage: CKPT=<...>/model_state_100.pt TARGET=ADRB2 sbatch --export=ALL scripts_lane/step0.sbatch
#SBATCH --job-name=laneA_step0
#SBATCH --partition=egroup,mgroup_sp,mgroup_ct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/ARRE/logs/%j.step0.out
#SBATCH --error=/path/to/ARRE/logs/%j.step0.err
set -euo pipefail
C=/path/to/ARRE
CODE=${CODE:-$C/lanes/A_pansub}
ENVP=$C/envs/cgflow
source /path/to/conda/etc/profile.d/conda.sh
conda activate "$ENVP"
export LD_LIBRARY_PATH="$ENVP/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline OMP_NUM_THREADS=4
export TMPDIR=/tmp/laneA_step0_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
: "${CKPT:?set CKPT to a model_state_<it>.pt}"
TARGET=${TARGET:-ADRB2}
OUT=${OUT:-$CODE/docs/lane_a/step0}
echo "host=$(hostname) $(date -Is) TARGET=$TARGET CKPT=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import cgflow;print('cgflow from',cgflow.__file__)"
python "$CODE/scripts_lane/step0_diag.py" --target "$TARGET" --ckpt "$CKPT" --out "$OUT" \
  --num_states "${NUM_STATES:-64}" --analyse "${ANALYSE:-48}" --reps "${REPS:-48}" --K "${K:-1024}" \
  --marg_reps "${MARG_REPS:-200}" --mc_actions "${MC_ACTIONS:-2048}" --behaviour "${BEHAVIOUR:-1}"
echo "step0 done $(date -Is)"
