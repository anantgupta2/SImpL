cd ~/scratch/SImpL

# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen2.5-32B-Instruct
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh meta-llama/Llama-3.1-8B-Instruct
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B-Base

# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen2.5-32B-Instruct
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B-Base



# export DATASET_NAME="race-c"
# export CHECKPOINT_ROOT="oat-output/race-c"
# # export RUN_DIR="oat-output/race-c/Qwen3-4B-Base_cot-only_0520T23:53:45"
# # sbatch --export=ALL,DATASET_NAME="race-c",CHECKPOINT_ROOT="oat-output/race-c",RUN_DIR="oat-output/race-c/Qwen3-4B-Base_cot-only_0520T23:53:45" scripts/eval/run_eval_final_saved_model.sh
# export RUN_DIR="oat-output/race-c/Qwen3-4B-Base_cot-only_0521T19:04:24"
# sbatch --export=ALL,DATASET_NAME="race-c",CHECKPOINT_ROOT="oat-output/race-c",RUN_DIR="oat-output/race-c/Qwen3-4B-Base_cot-only_0521T19:04:24" scripts/eval/run_eval_final_saved_model.sh

# sbatch --dependency=afterok:9189994 --export=ALL,DATASET_NAME="race-c",IS_INSTRUCT=1 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Instruct_cot-only_0526T20:27:19
# sbatch --dependency=afterok:9189994 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Instruct_cot-only_0526T20:27:19

# sbatch --dependency=afterok:9189995 --export=ALL,DATASET_NAME="race-c",IS_INSTRUCT=1 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base_cot-only_0526T20:27:19
# sbatch --dependency=afterok:9189995 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base_cot-only_0526T20:27:19

sbatch --dependency=afterok:9290021 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base_understanding-only_0528T21:37:14
sbatch --dependency=afterok:9290022 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-baseline_understanding-only_0528T21:37:36
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-8B-Base-trl-cot-only-0521_T23
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-8B-Base_cot-only_0521T23:33:31
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-trl-cot-only-0522_T03
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-trl-cot-only-0522_T03
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B_cot-only_0522T13:40:39
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-8B_cot-only_0522T13:42:04

# sbatch --dependency=afterok:9046776 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Instruct_understanding-only_0523T11:17:12