#!/bin/bash
#SBATCH -A aqu2_lab_gpu
#SBATCH -J diffts_sample
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --error=logs/diffts_sample-%J.err
#SBATCH --output=logs/diffts_sample-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

module purge
module load anaconda/2024.06
eval "$(conda shell.bash hook)"
conda activate diffts

cd ~/Diffusion-TS || { echo "ERROR: Could not cd to ~/Diffusion-TS"; exit 1; }
mkdir -p logs

echo "=========================================="
echo "Diffusion-TS Sampling - All Datasets"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo "=========================================="

OVERALL_EXIT=0

# Dataset configs and their checkpoint directories
declare -A CONFIGS
CONFIGS[stocks]="Config/stocks.yaml"
CONFIGS[etth]="Config/etth.yaml"
CONFIGS[energy]="Config/energy.yaml"
CONFIGS[fmri]="Config/fmri.yaml"
CONFIGS[sines]="Config/sines.yaml"
CONFIGS[mujoco]="Config/mujoco.yaml"
CONFIGS[mhealth]="Config/mhealth.yaml"

declare -A CKPT_DIRS
CKPT_DIRS[stocks]="Checkpoints_stock"
CKPT_DIRS[etth]="Checkpoints_etth"
CKPT_DIRS[energy]="Checkpoints_energy"
CKPT_DIRS[fmri]="Checkpoints_fmri"
CKPT_DIRS[sines]="Checkpoints_sines"
CKPT_DIRS[mujoco]="Checkpoints_mujoco"
CKPT_DIRS[mhealth]="Checkpoints_mhealth"

DATASETS="${DATASETS:-stocks;etth;energy;fmri;sines;mujoco;mhealth}"

IFS=';' read -ra DATASET_LIST <<< "$DATASETS"
for dataset in "${DATASET_LIST[@]}"; do
    config="${CONFIGS[$dataset]}"
    ckpt_dir="${CKPT_DIRS[$dataset]}"

    if [ -z "$config" ] || [ ! -f "$config" ]; then
        echo "SKIP: No config for $dataset"
        continue
    fi

    if [ ! -d "$ckpt_dir" ]; then
        echo "SKIP: No checkpoint directory $ckpt_dir for $dataset"
        continue
    fi

    # Find the latest milestone number from checkpoint files
    LATEST=$(ls "$ckpt_dir"/model-*.pt 2>/dev/null | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    if [ -z "$LATEST" ]; then
        echo "SKIP: No checkpoints found in $ckpt_dir for $dataset"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Sampling: $dataset (milestone=$LATEST)"
    echo "Config:   $config"
    echo "Started:  $(date)"
    echo "=========================================="

    python main.py \
        --name "$dataset" \
        --config_file "$config" \
        --gpu 0 \
        --milestone "$LATEST"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: $dataset sampling exited with code $EXIT_CODE"
        OVERALL_EXIT=1
    else
        echo "$dataset sampling completed successfully."
        # Show output file
        ls -la "OUTPUT/$dataset/ddpm_fake_${dataset}.npy" 2>/dev/null
    fi

    echo "Finished: $(date)"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All sampling completed at: $(date)"
echo "Exit code: $OVERALL_EXIT"
echo "=========================================="

exit $OVERALL_EXIT
