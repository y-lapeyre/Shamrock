#!/bin/bash
#SBATCH --account=<project account>
#SBATCH --job-name=<jobname>
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.out   # Same file for both!


module purge
source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh
# A CrayPE environment version
module load cpe/24.11
# An architecture
module load craype-accel-amd-gfx90a craype-x86-trento
# A compiler to target the architecture
module load PrgEnv-cray
# Some architecture related libraries and tools
module load amd-mixed/6.4.3

module list

export MPICH_GPU_SUPPORT_ENABLED=1
export SYCL_DEVICE_ALLOCATOR=aligned
# export OMP_<ICV=XXX>
WORKDIR=<path-to-Shamrock-build-directory>

. /opt/cray/pe/cpe/24.11/restore_lmod_system_defaults.sh

export SHAM_MAX_ALLOC_SIZE=4294967296

. ./activate

srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpus-per-task=1 --gpu-bind=closest -- ./shamrock --smi-full --sycl-cfg auto:HIP --force-dgpu-off --loglevel 1 --rscript runscript.py
