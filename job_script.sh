#!/bin/bash
#SBATCH -N 2
#SBATCH --cores-per-socket=10
#SBATCH --partition=batch
#SBATCH -J eliasOmega
#SBATCH -o eliasOmega.%J.out
#SBATCH -e eliasOmega.%J.err
#SBATCH --mail-user=shashwat.banchhor@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=36000
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4

module load machine_learning/2019.01-cudnn7.6-cuda10.0-py3.6
MASTER=${SLURMD_NODENAME}
# launch on process 2
python /ibex/scratch/banchhs/CIFAR10-dawn/dawn.py --master_address="tcp://"${MASTER}":9090" --rank=1 --world_size=${SLURM_JOB_NUM_NODES} &
# launch on process 1
python /ibex/scratch/banchhs/CIFAR10-dawn/dawn.py --master_address="tcp://"${MASTER}":9090" --rank=0 --world_size=${SLURM_JOB_NUM_NODES}


