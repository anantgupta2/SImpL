cd ~/scratch/SImpL
# sbatch scripts/run/cot_trl.sh Qwen4B-Instruct-race
# sbatch scripts/run/cot_trl.sh Qwen8B-Instruct-race
# sbatch scripts/run/cot_trl.sh Qwen8B-Base-race
# sbatch scripts/run/cot_oat.sh Qwen4B-Instruct-race
# sbatch scripts/run/cot_oat.sh Qwen4B-Base-race
# sbatch scripts/run/cot_oat.sh Qwen8B-Base-race

sbatch scripts/run/understanding_oat.sh Qwen4B-Base-race
sbatch scripts/run/understanding_oat.sh Qwen4B-Base-race-baseline