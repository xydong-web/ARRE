# Source this to get the shared cgflow env with the lane-D worktree first on PYTHONPATH.
#   source /path/to/ARRE/lanes/D_gate/scripts_lane/env.sh
LANE_D_ROOT=/path/to/ARRE/lanes/D_gate
source /path/to/conda/etc/profile.d/conda.sh
conda activate /path/to/ARRE/envs/cgflow
export LD_LIBRARY_PATH="/path/to/ARRE/envs/cgflow/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LANE_D_ROOT/src:$LANE_D_ROOT/pydeps:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
# Lane A: shared environment for tests / scripts.   usage:  source scripts_lane/env.sh
C=/path/to/ARRE
CODE=${CODE:-$C/lanes/A_pansub}
conda activate "$C/envs/cgflow"
export LD_LIBRARY_PATH="$C/envs/cgflow/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE/src:$CODE:$CODE/pydeps:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 WANDB_MODE=offline
