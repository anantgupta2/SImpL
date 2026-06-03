cd ~/scratch/SImpL

# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen2.5-32B-Instruct
# sbatch --export=ALL,DATASET_NAME="lsat-ar",IS_INSTRUCT=1 scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B
# sbatch --export=ALL,DATASET_NAME="lsat-ar",IS_INSTRUCT=1 scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base
# sbatch --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B-Base


# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen2.5-32B-Instruct
# sbatch --export=ALL,DATASET_NAME="race-c",IS_INSTRUCT=1 scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B
# sbatch --export=ALL,DATASET_NAME="race-c",IS_INSTRUCT=1 scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_untrained.sh Qwen/Qwen3-8B-Base



# sbatch --dependency=afterok:9189994 --export=ALL,DATASET_NAME="race-c",IS_INSTRUCT=1 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Instruct_cot-only_0526T20:27:19
# sbatch --dependency=afterok:9189994 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Instruct_cot-only_0526T20:27:19

# sbatch --dependency=afterok:9189995 --export=ALL,DATASET_NAME="race-c",IS_INSTRUCT=1 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base_cot-only_0526T20:27:19
# sbatch --dependency=afterok:9189995 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base_cot-only_0526T20:27:19

# sbatch --dependency=afterany:9392784 --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-200-samples_cot-only_42_0602T12:23:55
# sbatch --dependency=afterany:9392785 --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-200-samples_understanding-only_42_0602T12:36:31
# sbatch --dependency=afterany:9392786 --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-200-samples_combined-oat_42_0602T12:50:42
# sbatch --dependency=afterany:9392717 --export=ALL,DATASET_NAME="lsat-ar" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0602T12:18:46
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl_simpl-oat_42_0602T12:19:13
# sbatch --dependency=afterok:9383081 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl_simpl-oat_42_0601T19:23:50
# sbatch --dependency=afterok:9318742 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-baseline_combined-oat_0529T17:06:13
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-8B-Base-trl-cot-only-0521_T23
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-8B-Base_cot-only_0521T23:33:31
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-trl-cot-only-0522_T03
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-trl-cot-only-0522_T03
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B_cot-only_0522T13:40:39
# sbatch --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-8B_cot-only_0522T13:42:04

# sbatch --dependency=afterok:9046776 --export=ALL,DATASET_NAME="race-c" scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Instruct_understanding-only_0523T11:17:12