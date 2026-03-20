#!/bin/bash
#SBATCH -A aqu2_lab_gpu
#SBATCH -J diffts_mhealth
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --error=logs/diffts_mhealth-%J.err
#SBATCH --output=logs/diffts_mhealth-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

module purge
module load anaconda/2024.06
eval "$(conda shell.bash hook)"
conda activate diffts

cd ~/Diffusion-TS || { echo "ERROR: Could not cd to ~/Diffusion-TS"; exit 1; }
mkdir -p logs

echo "=========================================="
echo "Diffusion-TS Training - MHEALTH"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo "=========================================="

python main.py --name mhealth --config_file Config/mhealth.yaml --gpu 0 --train

echo ""
echo "Completed at: $(date)"
echo "Exit code: $?"
