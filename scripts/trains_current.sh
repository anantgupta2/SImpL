cd ~/scratch/SImpL
# sbatch scripts/run/cot_trl.sh Qwen4B-Instruct-race
# sbatch scripts/run/cot_trl.sh Qwen8B-Instruct-race
# sbatch scripts/run/cot_trl.sh Qwen8B-Base-race
# sbatch scripts/run/cot_oat.sh Qwen4B-Instruct-race
# sbatch scripts/run/cot_oat.sh Qwen4B-Base-race
# sbatch scripts/run/cot_oat.sh Qwen8B-Base-race

# sbatch scripts/run/cot_oat.sh Qwen4B-Base-race-baseline
# sbatch scripts/run/cot_oat.sh Qwen4B-Base-lsat-200
# sbatch scripts/run/understanding_oat.sh Qwen4B-Base-lsat-200
# sbatch --time=11:00:00 scripts/run/combined_oat.sh Qwen4B-Base-lsat-200
# sbatch scripts/run/simpl_oat.sh Qwen4B-Base-lsat-simpl

# sbatch scripts/run/simpl_oat.sh Qwen4B-Base-race-simpl

# sbatch scripts/run/understanding_oat.sh Qwen4B-Base-race-baseline 24
# sbatch scripts/run/cot_oat.sh Qwen4B-Base-race-baseline 24
# sbatch scripts/run/combined_oat.sh Qwen4B-Base-race-baseline 24

# sbatch scripts/run/understanding_oat.sh Qwen4B-Base-race-baseline 36
# sbatch scripts/run/cot_oat.sh Qwen4B-Base-race-baseline 36
# sbatch scripts/run/combined_oat.sh Qwen4B-Base-race-baseline 36