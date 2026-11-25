#!/usr/bin/env bash
# Lane B test suite on a GPU node: unit tests (T1/T2/T3) + the integration test (T4) that needs the
# block env, the pose model and UniDock.
#   sbatch /path/to/ARRE/lanes/B_replay/scripts_lane/test_lane_b_gpu.sbatch
# Optional env: CGFLOW_TEST_ENV_DIR (default data/smoke/env20k), CGFLOW_TEST_CKPT (default weights/smoke_stateflow.ckpt)
#SBATCH --job-name=laneB_test
#SBATCH --partition=egroup,mgroup_sp,mgroup_ct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/ARRE/logs/%j.laneB_test.out
#SBATCH --error=/path/to/ARRE/logs/%j.laneB_test.err
set -euo pipefail
C=/path/to/ARRE
CODE=$C/lanes/B_replay
ENV=$C/envs/cgflow
source /path/to/conda/etc/profile.d/conda.sh
conda activate "$ENV"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline OMP_NUM_THREADS=4
export TMPDIR=/tmp/cgf_laneB_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export PYTHONPATH="$CODE/src:$CODE/pydeps:${PYTHONPATH:-}"
export CGFLOW_LANE_B_GPU_TEST=1
echo "host=$(hostname) $(date -Is) JOB=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import cgflow;print('cgflow from',cgflow.__file__)"
cd "$CODE"
"$CODE/pydeps/bin/pytest" tests/lane_b -v -p no:cacheprovider
echo "tests done $(date -Is)"
