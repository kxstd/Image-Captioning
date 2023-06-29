#!/bin/bash

#!/bin/bash

#SBATCH -p small
#SBATCH -n 4
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

python evaluate.py --prediction_file /lustre/home/acct-stu/stu151/image_captioning/experiments/baseline/resnet101_attention_b32_emd300_predictions.json  --reference_file /lustre/home/acct-stu/stu168/data/image_captioning/flickr8k/caption.txt --output_file result_baseline.txt