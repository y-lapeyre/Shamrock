#!/bin/bash
#SBATCH --job-name=examplejob       # Job name
#SBATCH --output=examplejob.o%j     # Name of stdout output file
#SBATCH --error=examplejob.e%j      # Name of stderr error file
#SBATCH --partition=boost_usr_prod  # Booster partition (add --qos=boost_qos_dbg for short tests)
#SBATCH --nodes=1                   # Total number of nodes
#SBATCH --ntasks-per-node=4         # 1 MPI rank per GPU
#SBATCH --cpus-per-task=8           # 32 cores per node / 4 ranks
#SBATCH --gres=gpu:4                # 4 A100 per node
#SBATCH --time=00:10:00             # Run time (hh:mm:ss)
#SBATCH --account=<project account> # Project for billing

echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo

SHAMROCK_BIN_PATH=<path to folder with binary>

RSCRIPT=<runscript>

# load the same modules as the build environment
source $SHAMROCK_BIN_PATH/activate

cat << EOF > select_gpu
#!/bin/bash

export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

export ACPP_DEBUG_LEVEL=0

srun --cpu-bind=cores -- \
    ./select_gpu $SHAMROCK_BIN_PATH/shamrock --sycl-cfg auto:CUDA --loglevel 1 --smi \
    --rscript $RSCRIPT
