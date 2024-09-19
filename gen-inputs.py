#!/usr/bin/env python3

# TODO: Change me; JUWELS values
CPUS_PER_NODE = 32
GPUS_PER_NODE = 4

def write_input(nodes, cells, out):
    raw = rf"""{{
    "name": "run_nodes-{nodes}_cells-{cells}",
    "num-cells": {cells*nodes*GPUS_PER_NODE},
    "synapses": 10,
    "min-delay": 5,
    "duration": 200,
    "ring-size": 4,
    "dt": 0.025,
    "depth": 10,
    "complex": true,
    "branch-probs": [
        1,
        0.5
    ],
    "compartments": [
        20,
        2
    ],
    "lengths": [
        200,
        20
    ]
}}
"""
    with open(out, 'w') as fd:
        print(raw, file=fd)

def write_sbatch(nodes, cells, inp, out):
    raw = rf"""#!/bin/bash -x
#SBATCH --job-name="busyring-cells-{cells}-nodes-{nodes}"
#SBATCH --mail-user=
#SBATCH --mail-type=NONE
#SBATCH --nodes={nodes}
#SBATCH --ntasks={GPUS_PER_NODE*nodes}
#SBATCH --cpus-per-task={CPUS_PER_NODE//GPUS_PER_NODE}
#SBATCH --time=90
#SBATCH --output=job-cells-{cells}-nodes-{nodes}.out
#SBATCH --error=job-cells-{cells}-nodes-{nodes}.err
#SBATCH --gres=gpu:{GPUS_PER_NODE}
#SBATCH --account=try24_Arbor
#SBATCH --partition=boost_usr_prod
#SBATCH --exclusive

ml cuda gcc openmpi

srun busyring {inp}
    """
    with open(out, 'w') as fd:
        print(raw, file=fd)

for nodes in [1, 2, 4, 8, 16, 32, 64]:
    for cells in [64000]:
        json = f"input-cells-{cells}-nodes-{nodes}.json"
        job  = f"submit-cells-{cells}-nodes-{nodes}.job"
        write_input(nodes, cells, out=json)
        write_sbatch(nodes, cells, inp=json, out=job)
