#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J cdawn
#SBATCH -o cdawn.%J.out
#SBATCH -e cdawn.%J.err
#SBATCH --mail-user=shashwatbanchhor@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=01:30:00
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100]



function ibex_mapping {
        # route to wich code are we going to run
        CODE=dawn.py

        python -m torch.distributed.launch --master_addr="${IP}" --master_port="${PORT}" --node_rank=${RANK} --nnodes=${WORLD_SIZE} --nproc_per_node=1  \
        ${CODE} -j=6 \
       --world-size=${WORLD} \
        --dist-url="${IP}:${PORT}"  \
        --rank=${RANK} 
}

function ibex_setup {
        # Compatibility code with ibex
        export LC_ALL=en_US.utf8
        export WORLD_SIZE=$WORLD_SIZE
        export WORLD=$WORLD_SIZE
        export RANK=$SLURM_PROCID


        # What this does is it writes the ip of the master node i.e. the node with rank 0
        # into a file that everyone shares, the file goes into the `tmps` folder and has a unique name
        master_ip_file=tmps/${NAME}_ip.txt
        if [ "$RANK" ==  "0" ]
        then
             IP=( $(host $(hostname)) )
             IP=${IP[3]}
             #ifconfig ib0 | awk '/inet / {print $2}' > ip.txt
             echo $IP > ${master_ip_file}
        fi

        IP=( $(host $(hostname)) )
        IP=${IP[3]}
        echo $IP > "ip_${RANK}_.txt"

        # The sleep is to make sure the master had time to write to the file,
        # maybe 10 seconds is excesive but it's just to be sure
        sleep 10

        export IP=`cat ${master_ip_file}`
        # End of Compatibility code with ibex
}



#run the application:
# cdawn

echo "starting"
ibex_setup
module load machine_learning/2019.01-cudnn7.6-cuda10.0-py3.6
pip install --user torchvision==0.2.1
srun python dawn.py