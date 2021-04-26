#!/bin/bash
#BSUB -J my_solution_2
#BSUB -q gpu_v100
#BSUB -m "gpu19"
#BSUB -gpu num=1
#BSUB -outdir "./output"
#BSUB -o ./output/%J.out -e ./output/%J.err
python main.py
