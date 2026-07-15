# Experiment ledger (Claude runs)

Auto-appended by scripts/claude/submit.sh. One row per submitted job.

| submitted | WS | job id | purpose | command |
|---|---|---|---|---|
| 2026-06-03 14:14:37 | WS1 | 9411110 | CoT-only HP: lr=4e-6 seed42 (dev-select, default) | `sbatch scripts/run/cot_oat.sh hp/cot-lr4e6 42 qwen` |
| 2026-06-03 14:17:04 | WS1 | 9411148 | CoT-only HP: lr=8e-6 seed42 (dev-select) | `sbatch scripts/run/cot_oat.sh hp/cot-lr8e6 42 qwen` |
| 2026-06-03 14:22:28 | WS1 | 9411184 | dev-eval CoT lr=4e-6 (dep afterok:9411110) | `sbatch --dependency=afterok:9411110 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr4e6_cot-only_42_0603T14:15:14` |
| 2026-06-03 14:22:28 | WS1 | 9411185 | dev-eval CoT lr=8e-6 (dep afterok:9411148) | `sbatch --dependency=afterok:9411148 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr8e6_cot-only_42_0603T14:17:31` |
| 2026-06-03 14:25:43 | WS1 | 9411204 | CoT-only HP: lr=2e-6 seed42 (dev-select, 3rd LR point) | `sbatch scripts/run/cot_oat.sh hp/cot-lr2e6 42 qwen` |
| 2026-06-03 14:25:43 | WS1 | 9411205 | CoT-only HP: group=16 (reasoning_num_samples) lr=4e-6 seed42 (dev-select) | `sbatch scripts/run/cot_oat.sh hp/cot-g16 42 qwen` |
| 2026-06-03 14:26:26 | WS1 | 9411211 | dev-eval Qwen3-4B-Base-cot-hp-lr2e6_cot-only_42 (dep afterok:9411204) | `sbatch --dependency=afterok:9411204 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr2e6_cot-only_42_0603T14:26:11` |
| 2026-06-03 14:26:56 | WS1 | 9411214 | dev-eval Qwen3-4B-Base-cot-hp-g16_cot-only_42 (dep afterok:9411205) | `sbatch --dependency=afterok:9411205 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-g16_cot-only_42_0603T14:26:34` |
| 2026-06-03 18:41:43 | WS1 | 9415626 | CoT-only HP: lr=1.2e-5 seed42 (dev-select, extend up) | `sbatch scripts/run/cot_oat.sh hp/cot-lr12e6 42 qwen` |
| 2026-06-03 18:41:43 | WS1 | 9415627 | CoT-only HP: lr=1.6e-5 seed42 (dev-select, extend up) | `sbatch scripts/run/cot_oat.sh hp/cot-lr16e6 42 qwen` |
| 2026-06-03 19:10:08 | WS1 | 9416097 | CoT-only HP: lr=1.2e-5 seed42 (RESUBMIT; prev 9415626 wedged) | `sbatch scripts/run/cot_oat.sh hp/cot-lr12e6 42 qwen` |
| 2026-06-03 19:10:08 | WS1 | 9416098 | CoT-only HP: lr=1.6e-5 seed42 (RESUBMIT; prev 9415627 wedged) | `sbatch scripts/run/cot_oat.sh hp/cot-lr16e6 42 qwen` |
| 2026-06-03 19:12:06 | WS1 | 9416105 | dev-eval Qwen3-4B-Base-cot-hp-lr12e6_cot-only_42 (dep afterok:9416097) | `sbatch --dependency=afterok:9416097 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr12e6_cot-only_42_0603T19:11:58` |
| 2026-06-03 19:36:20 | WS1 | 9416235 | dev-eval CoT lr=1.6e-5 (dep afterok:9416098) | `sbatch --dependency=afterok:9416098 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr16e6_cot-only_42_0603T19:17:33` |
| 2026-06-03 21:17:07 | WS3 | 9417034 | CoT-only MAIN lr=8e-6 seed=42 (test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6 42 qwen` |
| 2026-06-03 21:17:07 | WS3 | 9417035 | CoT-only MAIN lr=8e-6 seed=24 (test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6 24 qwen` |
| 2026-06-03 21:17:07 | WS3 | 9417036 | CoT-only MAIN lr=8e-6 seed=36 (test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6 36 qwen` |
| 2026-06-03 21:17:07 | WS3 | 9417037 | SImpL(--simpl) MAIN lr=8e-6 seed=42 (test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6 42 qwen` |
| 2026-06-03 21:17:08 | WS3 | 9417038 | SImpL(--simpl) MAIN lr=8e-6 seed=24 (test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6 24 qwen` |
| 2026-06-03 21:17:08 | WS3 | 9417039 | SImpL(--simpl) MAIN lr=8e-6 seed=36 (test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6 36 qwen` |
| 2026-06-03 21:19:44 | WS3 | 9417058 | test-eval Qwen3-4B-Base-cot-main-lr8e6_cot-only_42 (dep afterok:9417034) | `sbatch --dependency=afterok:9417034 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6_cot-only_42_0603T21:19:01` |
| 2026-06-03 21:20:44 | WS3 | 9417066 | test-eval Qwen3-4B-Base-cot-main-lr8e6_cot-only_24 (dep afterok:9417035) | `sbatch --dependency=afterok:9417035 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6_cot-only_24_0603T21:20:05` |
| 2026-06-03 21:20:45 | WS3 | 9417067 | test-eval Qwen3-4B-Base-cot-main-lr8e6_cot-only_36 (dep afterok:9417036) | `sbatch --dependency=afterok:9417036 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6_cot-only_36_0603T21:20:27` |
| 2026-06-03 21:20:45 | WS3 | 9417068 | test-eval Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_42 (dep afterok:9417037) | `sbatch --dependency=afterok:9417037 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_42_0603T21:20:27` |
| 2026-06-03 21:21:45 | WS3 | 9417073 | test-eval Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_24 (dep afterok:9417038) | `sbatch --dependency=afterok:9417038 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_24_0603T21:21:28` |
| 2026-06-03 21:28:45 | WS3 | 9417093 | test-eval Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_36 (dep afterok:9417039) | `sbatch --dependency=afterok:9417039 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_36_0603T21:28:17` |
| 2026-06-04 10:39:34 | WS-DATAEFF | 9421886 | CoT-only n=50 lr=8e-6 seed=42 (data-eff curve, test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6-n50 42 qwen` |
| 2026-06-04 10:39:48 | WS-DATAEFF | 9421887 | CoT-only n=50 lr=8e-6 seed=24 (data-eff curve, test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6-n50 24 qwen` |
| 2026-06-04 10:39:48 | WS-DATAEFF | 9421888 | CoT-only n=50 lr=8e-6 seed=36 (data-eff curve, test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6-n50 36 qwen` |
| 2026-06-04 10:39:48 | WS-DATAEFF | 9421889 | SImpL n=50 lr=8e-6 seed=42 (data-eff curve, test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-n50 42 qwen` |
| 2026-06-04 10:39:49 | WS-DATAEFF | 9421890 | SImpL n=50 lr=8e-6 seed=24 (data-eff curve, test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-n50 24 qwen` |
| 2026-06-04 10:39:49 | WS-DATAEFF | 9421891 | SImpL n=50 lr=8e-6 seed=36 (data-eff curve, test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-n50 36 qwen` |
| 2026-06-04 10:43:16 | WS-DATAEFF | 9421927 | test-eval Qwen3-4B-Base-simpl-main-lr8e6-n50_simpl-oat_42 (dep afterok:9421889) | `sbatch --dependency=afterok:9421889 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-n50_simpl-oat_42_0604T10:42:22` |
| 2026-06-04 10:44:16 | WS-DATAEFF | 9421936 | test-eval Qwen3-4B-Base-simpl-main-lr8e6-n50_simpl-oat_24 (dep afterok:9421890) | `sbatch --dependency=afterok:9421890 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-n50_simpl-oat_24_0604T10:43:26` |
| 2026-06-04 10:44:16 | WS-DATAEFF | 9421937 | test-eval Qwen3-4B-Base-cot-main-lr8e6-n50_cot-only_24 (dep afterok:9421887) | `sbatch --dependency=afterok:9421887 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6-n50_cot-only_24_0604T10:44:15` |
| 2026-06-04 10:44:16 | WS-DATAEFF | 9421938 | test-eval Qwen3-4B-Base-cot-main-lr8e6-n50_cot-only_42 (dep afterok:9421886) | `sbatch --dependency=afterok:9421886 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6-n50_cot-only_42_0604T10:43:19` |
| 2026-06-04 10:44:17 | WS-DATAEFF | 9421939 | test-eval Qwen3-4B-Base-cot-main-lr8e6-n50_cot-only_36 (dep afterok:9421888) | `sbatch --dependency=afterok:9421888 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6-n50_cot-only_36_0604T10:43:30` |
| 2026-06-04 10:47:17 | WS-DATAEFF | 9421987 | test-eval Qwen3-4B-Base-simpl-main-lr8e6-n50_simpl-oat_36 (dep afterok:9421891) | `sbatch --dependency=afterok:9421891 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-n50_simpl-oat_36_0604T10:46:41` |
| 2026-06-04 10:49:58 | WS-DATAEFF | 9422092 | CoT-only n=100 lr=8e-6 seed=42 (data-eff curve, test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6-n100 42 qwen` |
| 2026-06-04 10:49:59 | WS-DATAEFF | 9422093 | CoT-only n=100 lr=8e-6 seed=24 (data-eff curve, test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6-n100 24 qwen` |
| 2026-06-04 10:49:59 | WS-DATAEFF | 9422094 | CoT-only n=100 lr=8e-6 seed=36 (data-eff curve, test eval) | `sbatch scripts/run/cot_oat.sh main/cot-lr8e6-n100 36 qwen` |
| 2026-06-04 10:49:59 | WS-DATAEFF | 9422095 | SImpL n=100 lr=8e-6 seed=42 (data-eff curve, test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-n100 42 qwen` |
| 2026-06-04 10:49:59 | WS-DATAEFF | 9422096 | SImpL n=100 lr=8e-6 seed=24 (data-eff curve, test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-n100 24 qwen` |
| 2026-06-04 10:49:59 | WS-DATAEFF | 9422097 | SImpL n=100 lr=8e-6 seed=36 (data-eff curve, test eval) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-n100 36 qwen` |
| 2026-06-04 10:53:21 | WS-DATAEFF | 9422219 | test-eval Qwen3-4B-Base-cot-main-lr8e6-n100_cot-only_42 (dep afterok:9422092) | `sbatch --dependency=afterok:9422092 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6-n100_cot-only_42_0604T10:53:07` |
| 2026-06-04 10:55:21 | WS-DATAEFF | 9422260 | test-eval Qwen3-4B-Base-cot-main-lr8e6-n100_cot-only_24 (dep afterok:9422093) | `sbatch --dependency=afterok:9422093 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6-n100_cot-only_24_0604T10:55:18` |
| 2026-06-04 11:09:22 | WS-DATAEFF | 9422837 | test-eval Qwen3-4B-Base-simpl-main-lr8e6-n100_simpl-oat_24 (dep afterok:9422096) | `sbatch --dependency=afterok:9422096 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-n100_simpl-oat_24_0604T11:08:30` |
| 2026-06-04 11:09:23 | WS-DATAEFF | 9422838 | test-eval Qwen3-4B-Base-cot-main-lr8e6-n100_cot-only_36 (dep afterok:9422094) | `sbatch --dependency=afterok:9422094 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-main-lr8e6-n100_cot-only_36_0604T11:08:30` |
| 2026-06-04 11:09:24 | WS-DATAEFF | 9422839 | test-eval Qwen3-4B-Base-simpl-main-lr8e6-n100_simpl-oat_42 (dep afterok:9422095) | `sbatch --dependency=afterok:9422095 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-n100_simpl-oat_42_0604T11:08:30` |
| 2026-06-04 11:11:24 | WS-DATAEFF | 9422866 | test-eval Qwen3-4B-Base-simpl-main-lr8e6-n100_simpl-oat_36 (dep afterok:9422097) | `sbatch --dependency=afterok:9422097 --export=ALL,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-n100_simpl-oat_36_0604T11:10:26` |
| 2026-06-04 11:47:50 | WS4-LSAT | 9423732 | CoT-only LSAT-AR n=100 lr=4e-6 seed=42 (test eval) | `sbatch scripts/run/cot_oat.sh Qwen4B-Base-lsat-100 42 qwen` |
| 2026-06-04 11:47:50 | WS4-LSAT | 9423733 | CoT-only LSAT-AR n=100 lr=4e-6 seed=24 (test eval) | `sbatch scripts/run/cot_oat.sh Qwen4B-Base-lsat-100 24 qwen` |
| 2026-06-04 11:47:50 | WS4-LSAT | 9423734 | CoT-only LSAT-AR n=100 lr=4e-6 seed=36 (test eval) | `sbatch scripts/run/cot_oat.sh Qwen4B-Base-lsat-100 36 qwen` |
| 2026-06-04 11:47:50 | WS4-LSAT | 9423735 | SImpL LSAT-AR n=100 lr=4e-6 seed=42 (logic prompt, test eval) | `sbatch scripts/run/simpl_oat.sh Qwen4B-Base-lsat-simpl 42 qwen` |
| 2026-06-04 11:47:50 | WS4-LSAT | 9423736 | SImpL LSAT-AR n=100 lr=4e-6 seed=24 (logic prompt, test eval) | `sbatch scripts/run/simpl_oat.sh Qwen4B-Base-lsat-simpl 24 qwen` |
| 2026-06-04 11:47:51 | WS4-LSAT | 9423737 | SImpL LSAT-AR n=100 lr=4e-6 seed=36 (logic prompt, test eval) | `sbatch scripts/run/simpl_oat.sh Qwen4B-Base-lsat-simpl 36 qwen` |
| 2026-06-04 11:50:14 | WS4-LSAT | 9423790 | test-eval Qwen3-4B-Base-100-samples_cot-only_42 (dep afterok:9423732) | `sbatch --dependency=afterok:9423732 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_42_0604T11:49:43` |
| 2026-06-04 11:50:14 | WS4-LSAT | 9423791 | test-eval Qwen3-4B-Base-100-samples_cot-only_24 (dep afterok:9423733) | `sbatch --dependency=afterok:9423733 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44` |
| 2026-06-04 11:55:15 | WS4-LSAT | 9424000 | test-eval Qwen3-4B-Base-100-samples_cot-only_36 (dep afterok:9423734) | `sbatch --dependency=afterok:9423734 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40` |
| 2026-06-04 11:58:15 | WS4-LSAT | 9424127 | test-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_42 (dep afterok:9423735) | `sbatch --dependency=afterok:9423735 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33` |
| 2026-06-04 11:58:15 | WS4-LSAT | 9424128 | test-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_24 (dep afterok:9423736) | `sbatch --dependency=afterok:9423736 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T11:57:59` |
| 2026-06-04 12:01:15 | WS4-LSAT | 9424231 | test-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_36 (dep afterok:9423737) | `sbatch --dependency=afterok:9423737 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33` |
| 2026-06-04 15:05:58 | WS5-XFER | 9428300 | race2lsat cross-eval: Qwen3-4B-Base-cot-main-lr8e6_cot-only_42_0603T21:19:01 | `sbatch --export=ALL,RUN_DIR=oat-output/race-c/Qwen3-4B-Base-cot-main-lr8e6_cot-only_42_0603T21:19:01,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/cross/race2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:05:58 | WS5-XFER | 9428301 | race2lsat cross-eval: Qwen3-4B-Base-cot-main-lr8e6_cot-only_24_0603T21:20:05 | `sbatch --export=ALL,RUN_DIR=oat-output/race-c/Qwen3-4B-Base-cot-main-lr8e6_cot-only_24_0603T21:20:05,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/cross/race2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:05:58 | WS5-XFER | 9428302 | race2lsat cross-eval: Qwen3-4B-Base-cot-main-lr8e6_cot-only_36_0603T21:20:27 | `sbatch --export=ALL,RUN_DIR=oat-output/race-c/Qwen3-4B-Base-cot-main-lr8e6_cot-only_36_0603T21:20:27,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/cross/race2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:05:58 | WS5-XFER | 9428303 | race2lsat cross-eval: Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_42_0603T21:20:27 | `sbatch --export=ALL,RUN_DIR=oat-output/race-c/Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_42_0603T21:20:27,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/cross/race2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:05:58 | WS5-XFER | 9428304 | race2lsat cross-eval: Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_24_0603T21:21:28 | `sbatch --export=ALL,RUN_DIR=oat-output/race-c/Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_24_0603T21:21:28,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/cross/race2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:05:59 | WS5-XFER | 9428305 | race2lsat cross-eval: Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_36_0603T21:28:17 | `sbatch --export=ALL,RUN_DIR=oat-output/race-c/Qwen3-4B-Base-simpl-main-lr8e6_simpl-oat_36_0603T21:28:17,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/cross/race2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:06:43 | WS5-XFER | 9428314 | lsat2race cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33 (dep afterok:9423737) | `sbatch --dependency=afterok:9423737 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:06:44 | WS5-XFER | 9428315 | lsat2race cross-eval Qwen3-4B-Base-100-samples_cot-only_42_0604T11:49:43 (dep afterok:9423732) | `sbatch --dependency=afterok:9423732 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_42_0604T11:49:43,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:06:44 | WS5-XFER | 9428316 | lsat2race cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33 (dep afterok:9423735) | `sbatch --dependency=afterok:9423735 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:06:44 | WS5-XFER | FAILED | lsat2race cross-eval Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40 (dep afterok:9423734) | `sbatch --dependency=afterok:9423734 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:06:44 | WS5-XFER | 9428318 | lsat2race cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T11:57:59 (dep afterok:9423736) | `sbatch --dependency=afterok:9423736 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T11:57:59,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:06:44 | WS5-XFER | FAILED | lsat2race cross-eval Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44 (dep afterok:9423733) | `sbatch --dependency=afterok:9423733 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:22:22 | WS5-XFER | 9428495 | lsat2race cross-eval Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44 (direct, train already COMPLETED) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 15:22:22 | WS5-XFER | 9428496 | lsat2race cross-eval Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40 (direct, train already COMPLETED) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 16:50:16 | WS4-LSAT | 9429736 | CoT LSAT-AR seed=42 RERUN (prev 9423732 CANCELLED at step_300) | `sbatch scripts/run/cot_oat.sh Qwen4B-Base-lsat-100 42 qwen` |
| 2026-06-04 16:50:16 | WS4-LSAT | 9429737 | SImpL LSAT-AR seed=24 RERUN (prev 9423736 CANCELLED at step_150) | `sbatch scripts/run/simpl_oat.sh Qwen4B-Base-lsat-simpl 24 qwen` |
| 2026-06-04 16:51:56 | WS4-LSAT | 9429766 | in-domain test-eval Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14 (dep afterok:9429736) | `sbatch --dependency=afterok:9429736 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14` |
| 2026-06-04 16:51:57 | WS5-XFER | 9429767 | lsat2race cross-eval Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14 (dep afterok:9429736) | `sbatch --dependency=afterok:9429736 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 16:53:57 | WS4-LSAT | 9429856 | in-domain test-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17 (dep afterok:9429737) | `sbatch --dependency=afterok:9429737 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17` |
| 2026-06-04 16:53:57 | WS5-XFER | 9429857 | lsat2race cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17 (dep afterok:9429737) | `sbatch --dependency=afterok:9429737 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17,DATASET_NAME=race-c,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 17:03:33 | WS6-RECLOR | 9430157 | ReClor base Qwen3-4B-Base zero-shot (find mid-range) | `sbatch --export=ALL,DATASET_NAME=reclor,DATA_PATH=data/reclor/test_42_all.jsonl,OUTPUT_DIR=evaluations/reclor/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 17:18:29 | WS6-PW | 9430613 | ProofWriter d2 base Qwen3-4B-Base zero-shot (pick mid-range depth) | `sbatch --export=ALL,DATASET_NAME=proofwriter-d2,DATA_PATH=data/proofwriter-d2/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d2/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 17:18:30 | WS6-PW | 9430615 | ProofWriter d3 base Qwen3-4B-Base zero-shot (pick mid-range depth) | `sbatch --export=ALL,DATASET_NAME=proofwriter-d3,DATA_PATH=data/proofwriter-d3/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d3/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 17:18:30 | WS6-PW | 9430616 | ProofWriter d5 base Qwen3-4B-Base zero-shot (pick mid-range depth) | `sbatch --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 17:28:58 | WS6-PW | 9430932 | ProofWriter d2 base 4B-Base zero-shot (PROPER T/F/U prompt) | `sbatch --export=ALL,DATASET_NAME=proofwriter-d2,DATA_PATH=data/proofwriter-d2/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d2/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 17:28:58 | WS6-PW | 9430933 | ProofWriter d3 base 4B-Base zero-shot (PROPER T/F/U prompt) | `sbatch --export=ALL,DATASET_NAME=proofwriter-d3,DATA_PATH=data/proofwriter-d3/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d3/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 17:28:58 | WS6-PW | 9430934 | ProofWriter d5 base 4B-Base zero-shot (PROPER T/F/U prompt) | `sbatch --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/baselines scripts/eval/run_eval_untrained.sh Qwen/Qwen3-4B-Base` |
| 2026-06-04 19:59:59 | WS6-PW | 9432613 | SImpL ProofWriter-d5 lr=8e-6 seed=42 (SMOKE/first; verify training path) | `sbatch scripts/run/simpl_oat.sh main/pw-d5-simpl 42 qwen` |
| 2026-06-04 20:02:56 | WS6-PW | 9432639 | CoT ProofWriter-d5 lr=8e-6 seed=42 (test eval) | `sbatch scripts/run/cot_oat.sh main/pw-d5-cot 42 qwen` |
| 2026-06-04 20:02:56 | WS6-PW | 9432640 | CoT ProofWriter-d5 lr=8e-6 seed=24 (test eval) | `sbatch scripts/run/cot_oat.sh main/pw-d5-cot 24 qwen` |
| 2026-06-04 20:02:56 | WS6-PW | 9432641 | CoT ProofWriter-d5 lr=8e-6 seed=36 (test eval) | `sbatch scripts/run/cot_oat.sh main/pw-d5-cot 36 qwen` |
| 2026-06-04 20:02:56 | WS6-PW | 9432642 | SImpL ProofWriter-d5 lr=8e-6 seed=24 (test eval) | `sbatch scripts/run/simpl_oat.sh main/pw-d5-simpl 24 qwen` |
| 2026-06-04 20:02:56 | WS6-PW | 9432643 | SImpL ProofWriter-d5 lr=8e-6 seed=36 (test eval) | `sbatch scripts/run/simpl_oat.sh main/pw-d5-simpl 36 qwen` |
| 2026-06-04 20:03:53 | WS6-PW | 9432648 | pw-d5 test-eval Qwen3-4B-Base-pw-d5-simpl_simpl-oat_24 (dep afterok:9432642) | `sbatch --dependency=afterok:9432642 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-simpl_simpl-oat_24_0604T20:03:29` |
| 2026-06-04 20:03:54 | WS6-PW | 9432650 | pw-d5 test-eval Qwen3-4B-Base-pw-d5-simpl_simpl-oat_42 (dep afterok:9432613) | `sbatch --dependency=afterok:9432613 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-simpl_simpl-oat_42_0604T20:00:46` |
| 2026-06-04 20:03:54 | WS6-PW | 9432651 | pw-d5 test-eval Qwen3-4B-Base-pw-d5-cot_cot-only_24 (dep afterok:9432640) | `sbatch --dependency=afterok:9432640 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-cot_cot-only_24_0604T20:03:30` |
| 2026-06-04 20:03:54 | WS6-PW | 9432652 | pw-d5 test-eval Qwen3-4B-Base-pw-d5-cot_cot-only_42 (dep afterok:9432639) | `sbatch --dependency=afterok:9432639 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-cot_cot-only_42_0604T20:03:31` |
| 2026-06-04 20:03:54 | WS6-PW | 9432653 | pw-d5 test-eval Qwen3-4B-Base-pw-d5-cot_cot-only_36 (dep afterok:9432641) | `sbatch --dependency=afterok:9432641 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-cot_cot-only_36_0604T20:03:29` |
| 2026-06-04 20:05:55 | WS6-PW | 9432662 | pw-d5 test-eval Qwen3-4B-Base-pw-d5-simpl_simpl-oat_36 (dep afterok:9432643) | `sbatch --dependency=afterok:9432643 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-simpl_simpl-oat_36_0604T20:05:04` |
| 2026-06-04 22:09:06 | WS4-LSAT | 9433472 | in-domain LSAT eval cot42 rerun (fix path-doubling) | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14` |
| 2026-06-04 22:09:06 | WS4-LSAT | 9433473 | in-domain LSAT eval simpl24 rerun (fix path-doubling) | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17` |
| 2026-06-04 22:16:08 | WS6-PW | 9433628 | pwd5->RACE Qwen3-4B-Base-pw-d5-cot_cot-only_42_0604T20:03:31 (st=RUNNING) | `sbatch --dependency=afterok:9432639 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-cot_cot-only_42_0604T20:03:31,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:08 | WS6-PW | 9433629 | pwd5->LSAT Qwen3-4B-Base-pw-d5-cot_cot-only_42_0604T20:03:31 (st=RUNNING) | `sbatch --dependency=afterok:9432639 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-cot_cot-only_42_0604T20:03:31,DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:09 | WS6-PW | 9433631 | pwd5->RACE Qwen3-4B-Base-pw-d5-cot_cot-only_24_0604T20:03:30 (st=RUNNING) | `sbatch --dependency=afterok:9432640 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-cot_cot-only_24_0604T20:03:30,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:10 | WS6-PW | 9433632 | pwd5->LSAT Qwen3-4B-Base-pw-d5-cot_cot-only_24_0604T20:03:30 (st=RUNNING) | `sbatch --dependency=afterok:9432640 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-cot_cot-only_24_0604T20:03:30,DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:11 | WS6-PW | 9433634 | pwd5->RACE Qwen3-4B-Base-pw-d5-cot_cot-only_36_0604T20:03:29 (st=RUNNING) | `sbatch --dependency=afterok:9432641 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-cot_cot-only_36_0604T20:03:29,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:11 | WS6-PW | 9433635 | pwd5->LSAT Qwen3-4B-Base-pw-d5-cot_cot-only_36_0604T20:03:29 (st=RUNNING) | `sbatch --dependency=afterok:9432641 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-cot_cot-only_36_0604T20:03:29,DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:11 | WS6-PW | 9433636 | pwd5->RACE Qwen3-4B-Base-pw-d5-simpl_simpl-oat_42_0604T20:00:46 (st=RUNNING) | `sbatch --dependency=afterok:9432613 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-simpl_simpl-oat_42_0604T20:00:46,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:12 | WS6-PW | 9433637 | pwd5->LSAT Qwen3-4B-Base-pw-d5-simpl_simpl-oat_42_0604T20:00:46 (st=RUNNING) | `sbatch --dependency=afterok:9432613 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-simpl_simpl-oat_42_0604T20:00:46,DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:12 | WS6-PW | 9433638 | pwd5->RACE Qwen3-4B-Base-pw-d5-simpl_simpl-oat_24_0604T20:03:29 (st=RUNNING) | `sbatch --dependency=afterok:9432642 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-simpl_simpl-oat_24_0604T20:03:29,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:12 | WS6-PW | 9433639 | pwd5->LSAT Qwen3-4B-Base-pw-d5-simpl_simpl-oat_24_0604T20:03:29 (st=RUNNING) | `sbatch --dependency=afterok:9432642 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-simpl_simpl-oat_24_0604T20:03:29,DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:12 | WS6-PW | 9433640 | pwd5->RACE Qwen3-4B-Base-pw-d5-simpl_simpl-oat_36_0604T20:05:04 (st=RUNNING) | `sbatch --dependency=afterok:9432643 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-simpl_simpl-oat_36_0604T20:05:04,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 22:16:13 | WS6-PW | 9433641 | pwd5->LSAT Qwen3-4B-Base-pw-d5-simpl_simpl-oat_36_0604T20:05:04 (st=RUNNING) | `sbatch --dependency=afterok:9432643 --export=ALL,RUN_DIR=oat-output/proofwriter-d5/Qwen3-4B-Base-pw-d5-simpl_simpl-oat_36_0604T20:05:04,DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/pwd5-2lsat scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 23:49:13 | WS6-PW | 9436502 | lsat2pwd5 cross-eval Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14 (direct) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/cross/lsat2pwd5 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 23:49:13 | WS6-PW | 9436503 | lsat2pwd5 cross-eval Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44 (direct) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/cross/lsat2pwd5 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 23:49:14 | WS6-PW | 9436504 | lsat2pwd5 cross-eval Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40 (direct) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/cross/lsat2pwd5 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 23:49:14 | WS6-PW | 9436505 | lsat2pwd5 cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33 (direct) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/cross/lsat2pwd5 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 23:49:14 | WS6-PW | 9436506 | lsat2pwd5 cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17 (direct) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/cross/lsat2pwd5 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-04 23:49:14 | WS6-PW | 9436507 | lsat2pwd5 cross-eval Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33 (direct) | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/cross/lsat2pwd5 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 09:41:43 | ANALYSIS | 9444637 | full-dump lsat2race Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/analysis/lsat2race_full,SAMPLE_COUNT=2000 scripts/claude/eval_full_dump.sh` |
| 2026-06-05 09:41:43 | ANALYSIS | 9444638 | full-dump lsat2race Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/analysis/lsat2race_full,SAMPLE_COUNT=2000 scripts/claude/eval_full_dump.sh` |
| 2026-06-05 09:41:43 | ANALYSIS | 9444639 | full-dump lsat2race Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/analysis/lsat2race_full,SAMPLE_COUNT=2000 scripts/claude/eval_full_dump.sh` |
| 2026-06-05 09:41:43 | ANALYSIS | 9444640 | full-dump lsat2race Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/analysis/lsat2race_full,SAMPLE_COUNT=2000 scripts/claude/eval_full_dump.sh` |
| 2026-06-05 09:41:43 | ANALYSIS | 9444641 | full-dump lsat2race Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/analysis/lsat2race_full,SAMPLE_COUNT=2000 scripts/claude/eval_full_dump.sh` |
| 2026-06-05 09:41:43 | ANALYSIS | 9444642 | full-dump lsat2race Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/analysis/lsat2race_full,SAMPLE_COUNT=2000 scripts/claude/eval_full_dump.sh` |
| 2026-06-05 10:18:27 | WS7-RATIO | 9445167 | SImpL RACE u_rows=3 lr=8e-6 seed=42 (ratio scout: up-weight understanding 3x) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-u3 42 qwen` |
| 2026-06-05 10:18:27 | WS7-RATIO | 9445168 | SImpL ProofWriter-d5 u_rows=3 lr=8e-6 seed=42 (ratio scout) | `sbatch scripts/run/simpl_oat.sh main/pw-d5-simpl-u3 42 qwen` |
| 2026-06-05 10:21:31 | WS7-RATIO | 9445290 | RACE u3 test-eval (afterok:9445167) | `sbatch --dependency=afterok:9445167 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-u3_simpl-oat_42_0605T10:19:44` |
| 2026-06-05 10:21:31 | WS7-RATIO | 9445291 | PW-d5 u3 test-eval (afterok:9445168) | `sbatch --dependency=afterok:9445168 --export=ALL,DATASET_NAME=proofwriter-d5,DATA_PATH=data/proofwriter-d5/test_42_300.jsonl,OUTPUT_DIR=evaluations/proofwriter-d5/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-pw-d5-simpl-u3_simpl-oat_42_0605T10:19:46` |
| 2026-06-05 10:29:26 | WS7-RATIO | 9445420 | SImpL LSAT 1:1 ratio (cot_rows=1,u_rows=1) lr=4e-6 seed=42 (test+transfer eval) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-1to1 42 qwen` |
| 2026-06-05 10:33:56 | WS7-RATIO | 9445513 | LSAT 1:1 in-domain test-eval (afterok:9445420) | `sbatch --dependency=afterok:9445420 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-lsat-simpl-1to1_simpl-oat_42_0605T10:30:48` |
| 2026-06-05 10:33:56 | WS7-RATIO | 9445514 | LSAT 1:1 -> RACE transfer eval (afterok:9445420) | `sbatch --dependency=afterok:9445420 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-lsat-simpl-1to1_simpl-oat_42_0605T10:30:48,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 10:47:37 | WS7-TEMPLATE | 9445838 | SImpL LSAT use_full_understanding_output=true seed=42 (template ablation vs strict tags) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-fullout 42 qwen` |
| 2026-06-05 10:50:07 | WS7-TEMPLATE | 9445886 | LSAT fullout in-domain test-eval (afterok:9445838) | `sbatch --dependency=afterok:9445838 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-lsat-simpl-fullout_simpl-oat_42_0605T10:49:45` |
| 2026-06-05 10:50:07 | WS7-TEMPLATE | 9445887 | LSAT fullout -> RACE transfer eval (afterok:9445838) | `sbatch --dependency=afterok:9445838 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-lsat-simpl-fullout_simpl-oat_42_0605T10:49:45,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:26:02 | WS7-RATIO | 9457161 | SImpL RACE u_rows=3 seed=42 RERUN (prev 9445167 hung at step 73) | `sbatch scripts/run/simpl_oat.sh main/simpl-lr8e6-u3 42 qwen` |
| 2026-06-05 15:27:13 | WS7-RATIO | 9457248 | RACE u3 RERUN test-eval (afterok:9457161) | `sbatch --dependency=afterok:9457161 --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/race-c/final_only scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-simpl-main-lr8e6-u3_simpl-oat_42_0605T15:26:35` |
| 2026-06-05 15:32:19 | WS-EVAL8 | 9457446 | LSAT in-domain avg@8 Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14` |
| 2026-06-05 15:32:19 | WS-EVAL8 | 9457447 | LSAT in-domain avg@8 Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44` |
| 2026-06-05 15:32:19 | WS-EVAL8 | 9457448 | LSAT in-domain avg@8 Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40` |
| 2026-06-05 15:32:19 | WS-EVAL8 | 9457449 | LSAT in-domain avg@8 Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33` |
| 2026-06-05 15:32:20 | WS-EVAL8 | 9457450 | LSAT in-domain avg@8 Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17` |
| 2026-06-05 15:32:20 | WS-EVAL8 | 9457451 | LSAT in-domain avg@8 Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33` |
| 2026-06-05 15:41:17 | WS-EVAL8 | 9457939 | lsat2race avg@8 Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_42_0604T16:51:14,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:41:17 | WS-EVAL8 | 9457941 | lsat2race avg@8 Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_24_0604T11:49:44,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:41:17 | WS-EVAL8 | 9457942 | lsat2race avg@8 Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples_cot-only_36_0604T11:54:40,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:41:20 | WS-EVAL8 | 9457944 | lsat2race avg@8 Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_42_0604T11:57:33,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:41:20 | WS-EVAL8 | 9457947 | lsat2race avg@8 Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_24_0604T16:53:17,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:41:21 | WS-EVAL8 | 9457949 | lsat2race avg@8 Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-100-samples-simpl_simpl-oat_36_0604T12:00:33,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:48:37 | WS-EVAL8 | 9458542 | fullout LSAT in-domain avg@8 seed42 | `sbatch --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-lsat-simpl-fullout_simpl-oat_42_0605T10:49:45` |
| 2026-06-05 15:48:37 | WS-EVAL8 | 9458544 | fullout LSAT->RACE transfer avg@8 seed42 | `sbatch --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-lsat-simpl-fullout_simpl-oat_42_0605T10:49:45,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 15:48:37 | WS7-TEMPLATE | 9458545 | SImpL LSAT fullout seed=24 (3-seed fullout vs strict) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-fullout 24 qwen` |
| 2026-06-05 15:48:37 | WS7-TEMPLATE | 9458546 | SImpL LSAT fullout seed=36 (3-seed fullout vs strict) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-fullout 36 qwen` |
| 2026-06-05 15:55:36 | WS7-TEMPLATE | 9459225 | SImpL LSAT fullout-NOTAG seed=42 (fixed untagged understanding prompt) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-fullout 42 qwen` |
| 2026-06-05 15:55:37 | WS7-TEMPLATE | 9459227 | SImpL LSAT fullout-NOTAG seed=24 (fixed untagged understanding prompt) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-fullout 24 qwen` |
| 2026-06-05 15:55:38 | WS7-TEMPLATE | 9459228 | SImpL LSAT fullout-NOTAG seed=36 (fixed untagged understanding prompt) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-fullout 36 qwen` |
| 2026-06-05 16:01:36 | WS7-TEMPLATE | 9459813 | fullout-notag in-domain avg@8 (afterok:9459225) | `sbatch --dependency=afterok:9459225 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-lsat-simpl-fullout-notag_simpl-oat_42_0605T15:56:08` |
| 2026-06-05 16:01:37 | WS7-TEMPLATE | 9459815 | fullout-notag ->RACE transfer avg@8 (afterok:9459225) | `sbatch --dependency=afterok:9459225 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-lsat-simpl-fullout-notag_simpl-oat_42_0605T15:56:08,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 16:01:37 | WS7-TEMPLATE | 9459816 | fullout-notag in-domain avg@8 (afterok:9459227) | `sbatch --dependency=afterok:9459227 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-lsat-simpl-fullout-notag_simpl-oat_24_0605T15:56:09` |
| 2026-06-05 16:01:37 | WS7-TEMPLATE | 9459817 | fullout-notag ->RACE transfer avg@8 (afterok:9459227) | `sbatch --dependency=afterok:9459227 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-lsat-simpl-fullout-notag_simpl-oat_24_0605T15:56:09,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 16:01:37 | WS7-TEMPLATE | 9459819 | fullout-notag in-domain avg@8 (afterok:9459228) | `sbatch --dependency=afterok:9459228 --export=ALL,DATASET_NAME=lsat-ar,OUTPUT_DIR=evaluations/lsat-ar/final_only_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-lsat-simpl-fullout-notag_simpl-oat_36_0605T15:56:13` |
| 2026-06-05 16:01:37 | WS7-TEMPLATE | 9459820 | fullout-notag ->RACE transfer avg@8 (afterok:9459228) | `sbatch --dependency=afterok:9459228 --export=ALL,RUN_DIR=oat-output/lsat-ar/Qwen3-4B-Base-lsat-simpl-fullout-notag_simpl-oat_36_0605T15:56:13,DATASET_NAME=race-c,DATA_PATH=data/race-c/test_42_all.jsonl,OUTPUT_DIR=evaluations/cross/lsat2race_avg8,COT_SAMPLES=8 scripts/eval/run_eval_final_saved_model.sh` |
| 2026-06-05 22:26:04 | WS-DRAWINGBOARD | 9478360 | SImpL-B LSAT use_understanding_passage=FALSE seed=42 (QA from understanding ALONE; attacks reward redundancy) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-nopassage 42 qwen` |
| 2026-06-05 22:26:04 | WS-DRAWINGBOARD | 9478361 | SImpL-B LSAT use_understanding_passage=FALSE seed=24 (QA from understanding ALONE; attacks reward redundancy) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-nopassage 24 qwen` |
| 2026-06-05 22:26:04 | WS-DRAWINGBOARD | 9478362 | SImpL-B LSAT use_understanding_passage=FALSE seed=36 (QA from understanding ALONE; attacks reward redundancy) | `sbatch scripts/run/simpl_oat.sh main/lsat-simpl-nopassage 36 qwen` |
| 2026-06-05 22:27:43 | WS-DRAWINGBOARD | 9478378 | SImpL-B nopassage in-domain avg@8 seed42 (afterok:9478360) | `eval ... final_only_avg8 COT_SAMPLES=8` |
| 2026-06-05 22:27:43 | WS-DRAWINGBOARD | 9478379 | SImpL-B nopassage ->RACE transfer avg@8 seed42 (afterok:9478360) | `eval ... lsat2race_avg8 COT_SAMPLES=8` |
| 2026-06-05 22:27:43 | WS-DRAWINGBOARD | 9478380 | SImpL-B nopassage in-domain avg@8 seed24 (afterok:9478361) | `eval ... final_only_avg8 COT_SAMPLES=8` |
| 2026-06-05 22:27:43 | WS-DRAWINGBOARD | 9478381 | SImpL-B nopassage ->RACE transfer avg@8 seed24 (afterok:9478361) | `eval ... lsat2race_avg8 COT_SAMPLES=8` |
| 2026-06-06 14:57:49 | WS-STAGED | 9508534 | Task2: RACE-50 understanding-only marginal, reward_scale=2 seed42 | `sbatch scripts/run/simpl_marginal_oat.sh main/race-uonly-marg-rs2 42 qwen` |
| 2026-06-06 14:57:49 | WS-STAGED | 9508546 | Task2: RACE-50 understanding-only marginal, reward_scale=4 seed42 | `sbatch scripts/run/simpl_marginal_oat.sh main/race-uonly-marg-rs4 42 qwen` |
| 2026-06-06 14:57:49 | WS-STAGED | 9508553 | Task2: RACE-50 understanding-only marginal, reward_scale=5 seed42 | `sbatch scripts/run/simpl_marginal_oat.sh main/race-uonly-marg-rs5 42 qwen` |
| 2026-06-06 15:14:08 | WS-STAGED | 9508756 | Task3 lsat staged cot->simpl-marginal seed42 (afterok:9508747) | `sbatch --dependency=afterok:9508747 scripts/run/simpl_marginal_oat.sh main/lsat-staged-simpl 42 qwen` |
| 2026-06-06 15:14:08 | WS-STAGED | 9508757 | Task3 lsat staged cot->cot seed42 (afterok:9508747) | `sbatch --dependency=afterok:9508747 scripts/run/cot_oat.sh main/lsat-staged-cot 42 qwen` |
| 2026-06-06 15:14:08 | WS-STAGED | 9508759 | Task3 lsat staged cot->uonly-marginal seed42 (afterok:9508747) | `sbatch --dependency=afterok:9508747 scripts/run/simpl_marginal_oat.sh main/lsat-staged-uonly 42 qwen` |
| 2026-06-06 15:14:08 | WS-STAGED | 9508760 | Task3 race staged cot->simpl-marginal seed42 (afterok:9508748) | `sbatch --dependency=afterok:9508748 scripts/run/simpl_marginal_oat.sh main/race-staged-simpl 42 qwen` |
| 2026-06-06 15:14:08 | WS-STAGED | 9508761 | Task3 race staged cot->cot seed42 (afterok:9508748) | `sbatch --dependency=afterok:9508748 scripts/run/cot_oat.sh main/race-staged-cot 42 qwen` |
| 2026-06-06 15:14:08 | WS-STAGED | 9508762 | Task3 race staged cot->uonly-marginal seed42 (afterok:9508748) | `sbatch --dependency=afterok:9508748 scripts/run/simpl_marginal_oat.sh main/race-staged-uonly 42 qwen` |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508804 | Task2 rs2 RACE in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508805 | Task2 rs4 RACE in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508806 | Task2 rs5 RACE in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508807 | Task3 lsat simpl in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508808 | Task3 lsat simpl ->RACE avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508809 | Task3 race simpl in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508810 | Task3 race simpl ->LSAT avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508811 | Task3 lsat cot in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508812 | Task3 lsat cot ->RACE avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508813 | Task3 race cot in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508814 | Task3 race cot ->LSAT avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508815 | Task3 lsat uonly in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508816 | Task3 lsat uonly ->RACE avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508817 | Task3 race uonly in-domain avg@8 | auto_eval_chain |
| 2026-06-06 15:16:56 | WS-STAGED-EVAL | 9508818 | Task3 race uonly ->LSAT avg@8 | auto_eval_chain |
| 2026-06-06 19:47:23 | WS-QADIRECT | 9514797 | direct-sweep think RACE-50 uonly (race-uonly-marg-rs2) | `simpl_marginal_oat.sh main/race-uonly-marg-rs2 42` |
| 2026-06-06 19:47:23 | WS-QADIRECT | 9514799 | direct-sweep letter RACE-50 uonly (race-uonly-direct-letter) | `simpl_marginal_oat.sh main/race-uonly-direct-letter 42` |
| 2026-06-06 19:47:23 | WS-QADIRECT | 9514801 | direct-sweep tiny RACE-50 uonly (race-uonly-direct-tiny) | `simpl_marginal_oat.sh main/race-uonly-direct-tiny 42` |
| 2026-06-06 19:50:48 | WS-QADIRECT | 9514851 | direct-sweep COT-TRAINED race-uonly-tiny-cottrained (RACE-50 uonly, warm-start) | `simpl_marginal_oat.sh main/race-uonly-tiny-cottrained 42` |
| 2026-06-06 19:50:48 | WS-QADIRECT | 9514853 | direct-sweep COT-TRAINED race-uonly-think-cottrained (RACE-50 uonly, warm-start) | `simpl_marginal_oat.sh main/race-uonly-think-cottrained 42` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519526 | cot-trained uonly think seed42 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-think-cottrained 42 qwen <base_seed42>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519528 | cot-trained uonly think seed24 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-think-cottrained 24 qwen <base_seed24>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519530 | cot-trained uonly think seed36 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-think-cottrained 36 qwen <base_seed36>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519532 | cot-trained uonly tiny seed42 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-tiny-cottrained 42 qwen <base_seed42>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519534 | cot-trained uonly tiny seed24 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-tiny-cottrained 24 qwen <base_seed24>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519536 | cot-trained uonly tiny seed36 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-tiny-cottrained 36 qwen <base_seed36>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519538 | cot-trained uonly letter seed42 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-letter-cottrained 42 qwen <base_seed42>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519540 | cot-trained uonly letter seed24 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-letter-cottrained 24 qwen <base_seed24>` |
| 2026-06-06 21:36:41 | WS-COTTRAINED | 9519542 | cot-trained uonly letter seed36 (RACE-100, warm-start) | `simpl_marginal_oat.sh main/race-uonly-letter-cottrained 36 qwen <base_seed36>` |
| 2026-06-06 21:37:07 | WS-COTTRAINED | 9519560 | RACE cot-only baseline avg@8 seed42 (Qwen3-4B-Base-cot-main-lr8e6_cot-only_42_0603T21:19:01) | run_eval COT_SAMPLES=8 |
| 2026-06-06 21:37:07 | WS-COTTRAINED | 9519561 | RACE cot-only baseline avg@8 seed24 (Qwen3-4B-Base-cot-main-lr8e6_cot-only_24_0603T21:20:05) | run_eval COT_SAMPLES=8 |
| 2026-06-06 21:37:07 | WS-COTTRAINED | 9519562 | RACE cot-only baseline avg@8 seed36 (Qwen3-4B-Base-cot-main-lr8e6_cot-only_36_0603T21:20:27) | run_eval COT_SAMPLES=8 |
| 2026-06-06 23:34:22 | WS-COTTRAINED | 9526911 | staged cot->simpl-marginal LETTER-16 seed42 (RACE-100, cot+und) | `simpl_marginal_oat.sh main/race-staged-simpl-letter 42 qwen <base_seed42>` |
| 2026-06-06 23:34:22 | WS-COTTRAINED | 9526915 | staged cot->simpl-marginal LETTER-16 seed24 (RACE-100, cot+und) | `simpl_marginal_oat.sh main/race-staged-simpl-letter 24 qwen <base_seed24>` |
| 2026-06-06 23:34:22 | WS-COTTRAINED | 9526919 | staged cot->simpl-marginal LETTER-16 seed36 (RACE-100, cot+und) | `simpl_marginal_oat.sh main/race-staged-simpl-letter 36 qwen <base_seed36>` |
| 2026-06-06 23:38:22 | WS-COTTRAINED | 9527115 | control: cot->MORE-COT seed42 (RACE-100, warm-start) | `cot_oat.sh main/race-cottrained-morecot 42 qwen <base>` |
| 2026-06-06 23:38:22 | WS-COTTRAINED | 9527119 | control: cot->MORE-COT seed24 (RACE-100, warm-start) | `cot_oat.sh main/race-cottrained-morecot 24 qwen <base>` |
| 2026-06-06 23:38:22 | WS-COTTRAINED | 9527130 | control: cot->MORE-COT seed36 (RACE-100, warm-start) | `cot_oat.sh main/race-cottrained-morecot 36 qwen <base>` |
| 2026-06-06 23:38:22 | WS-COTTRAINED | 9527136 | staged simpl 1:1 und:cot LETTER-16 seed42 (RACE-100) | `simpl_marginal_oat.sh main/race-staged-simpl-letter-1to1 42 qwen <base>` |
| 2026-06-06 23:38:22 | WS-COTTRAINED | 9527141 | staged simpl 1:1 und:cot LETTER-16 seed24 (RACE-100) | `simpl_marginal_oat.sh main/race-staged-simpl-letter-1to1 24 qwen <base>` |
| 2026-06-06 23:38:22 | WS-COTTRAINED | 9527145 | staged simpl 1:1 und:cot LETTER-16 seed36 (RACE-100) | `simpl_marginal_oat.sh main/race-staged-simpl-letter-1to1 36 qwen <base>` |
| 2026-06-07 10:13:20 | WS-LSATPORT | 9551779 | LSAT staged-simpl-letter seed42 (LSAT, warm-start) | `simpl_marginal_oat.sh main/lsat-staged-simpl-letter 42 qwen <base>` |
| 2026-06-07 10:13:20 | WS-LSATPORT | 9551785 | LSAT staged-simpl-letter seed24 (LSAT, warm-start) | `simpl_marginal_oat.sh main/lsat-staged-simpl-letter 24 qwen <base>` |
| 2026-06-07 10:13:20 | WS-LSATPORT | 9551790 | LSAT staged-simpl-letter seed36 (LSAT, warm-start) | `simpl_marginal_oat.sh main/lsat-staged-simpl-letter 36 qwen <base>` |
| 2026-06-07 10:13:20 | WS-LSATPORT | 9551793 | LSAT more-cot control seed42 (LSAT, warm-start) | `cot_oat.sh main/lsat-cottrained-morecot 42 qwen <base>` |
| 2026-06-07 10:13:20 | WS-LSATPORT | 9551795 | LSAT more-cot control seed24 (LSAT, warm-start) | `cot_oat.sh main/lsat-cottrained-morecot 24 qwen <base>` |
| 2026-06-07 10:13:20 | WS-LSATPORT | 9551797 | LSAT more-cot control seed36 (LSAT, warm-start) | `cot_oat.sh main/lsat-cottrained-morecot 36 qwen <base>` |
| 2026-06-07 11:12:19 | WS-SPICE | 9554532 | SPICE smoke-test seed42 RACE-100 (frontier curriculum, letter-16) | `simpl_spice_oat.sh main/race-spice-letter 42 qwen <base>` |
| 2026-06-07 11:32:41 | WS-SPICE | 9555500 | SPICE LSAT seed42 (frontier curriculum, letter-16, epochs=10) | `simpl_spice_oat.sh main/lsat-spice-letter 42 qwen <base>` |
| 2026-06-07 11:32:41 | WS-SPICE | 9555503 | SPICE LSAT seed24 (frontier curriculum, letter-16, epochs=10) | `simpl_spice_oat.sh main/lsat-spice-letter 24 qwen <base>` |
| 2026-06-07 11:32:41 | WS-SPICE | 9555505 | SPICE LSAT seed36 (frontier curriculum, letter-16, epochs=10) | `simpl_spice_oat.sh main/lsat-spice-letter 36 qwen <base>` |
| 2026-06-07 11:33:25 | WS-SPICE | 9555538 | SPICE LSAT seed42 (frontier curriculum, letter-16, epochs=7) | `simpl_spice_oat.sh main/lsat-spice-letter 42 qwen <base>` |
| 2026-06-07 11:33:25 | WS-SPICE | 9555540 | SPICE LSAT seed24 (frontier curriculum, letter-16, epochs=7) | `simpl_spice_oat.sh main/lsat-spice-letter 24 qwen <base>` |
| 2026-06-07 11:33:25 | WS-SPICE | 9555542 | SPICE LSAT seed36 (frontier curriculum, letter-16, epochs=7) | `simpl_spice_oat.sh main/lsat-spice-letter 36 qwen <base>` |
| 2026-06-07 13:07:56 | WS-LSATPORT | 9561272 | TRANSFER morecot->RACE avg@8 seed42 | auto_eval_chain |
| 2026-06-07 13:07:56 | WS-LSATPORT | 9561274 | TRANSFER morecot->RACE avg@8 seed24 | auto_eval_chain |
| 2026-06-07 13:07:56 | WS-LSATPORT | 9561278 | TRANSFER morecot->RACE avg@8 seed36 | auto_eval_chain |
| 2026-06-07 13:07:56 | WS-LSATPORT | 9561279 | TRANSFER staged->RACE avg@8 seed42 | auto_eval_chain |
| 2026-06-07 13:07:56 | WS-LSATPORT | 9561281 | TRANSFER staged->RACE avg@8 seed24 | auto_eval_chain |
| 2026-06-07 13:07:56 | WS-LSATPORT | 9561282 | TRANSFER staged->RACE avg@8 seed36 | auto_eval_chain |
| 2026-06-07 13:51:13 | WS-SPICE | 9564655 | SPICE LSAT in-domain avg@8 seed42 (final) | auto_eval_chain |
| 2026-06-07 13:51:13 | WS-SPICE | 9564658 | SPICE LSAT->RACE transfer avg@8 seed42 | auto_eval_chain |
| 2026-06-07 13:51:13 | WS-SPICE | 9564659 | SPICE LSAT in-domain avg@8 seed24 (final) | auto_eval_chain |
| 2026-06-07 13:51:13 | WS-SPICE | 9564660 | SPICE LSAT->RACE transfer avg@8 seed24 | auto_eval_chain |
| 2026-06-07 13:51:13 | WS-SPICE | 9564661 | SPICE LSAT in-domain avg@8 seed36 (final) | auto_eval_chain |
| 2026-06-07 13:51:13 | WS-SPICE | 9564662 | SPICE LSAT->RACE transfer avg@8 seed36 | auto_eval_chain |
| 2026-06-07 14:17:42 | WS-CMP | 9566342 | CMP cot-from-base seed42 (LSAT-100, epochs=7) | `cot_oat.sh main/lsat-cot-base 42` |
| 2026-06-07 14:17:42 | WS-CMP | 9566344 | CMP cot-from-base seed24 (LSAT-100, epochs=7) | `cot_oat.sh main/lsat-cot-base 24` |
| 2026-06-07 14:17:42 | WS-CMP | 9566346 | CMP cot-from-base seed36 (LSAT-100, epochs=7) | `cot_oat.sh main/lsat-cot-base 36` |
| 2026-06-07 14:17:42 | WS-CMP | 9566348 | CMP spice-from-base seed42 (LSAT-100, epochs=7) | `simpl_spice_oat.sh main/lsat-spice-base 42` |
| 2026-06-07 14:17:42 | WS-CMP | 9566350 | CMP spice-from-base seed24 (LSAT-100, epochs=7) | `simpl_spice_oat.sh main/lsat-spice-base 24` |
| 2026-06-07 14:17:42 | WS-CMP | 9566352 | CMP spice-from-base seed36 (LSAT-100, epochs=7) | `simpl_spice_oat.sh main/lsat-spice-base 36` |
| 2026-06-07 14:17:42 | WS-CMP | 9566354 | CMP cot-warm@7 seed42 (LSAT-100, epochs=7) | `cot_oat.sh main/lsat-cot-warm 42` |
| 2026-06-07 14:17:42 | WS-CMP | 9566356 | CMP cot-warm@7 seed24 (LSAT-100, epochs=7) | `cot_oat.sh main/lsat-cot-warm 24` |
| 2026-06-07 14:17:42 | WS-CMP | 9566358 | CMP cot-warm@7 seed36 (LSAT-100, epochs=7) | `cot_oat.sh main/lsat-cot-warm 36` |
| 2026-06-07 14:28:38 | WS-CMP | 9567182 | cot-from-base seed42 (LSAT-100,ep7) + curve 9567183 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567184 | cot-from-base seed24 (LSAT-100,ep7) + curve 9567186 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567187 | cot-from-base seed36 (LSAT-100,ep7) + curve 9567188 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567189 | spice-from-base-K4 seed42 (LSAT-100,ep7) + curve 9567190 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567192 | spice-from-base-K4 seed24 (LSAT-100,ep7) + curve 9567193 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567194 | spice-from-base-K4 seed36 (LSAT-100,ep7) + curve 9567195 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567197 | cot-warm@7 seed42 (LSAT-100,ep7) + curve 9567198 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567199 | cot-warm@7 seed24 (LSAT-100,ep7) + curve 9567200 | requeue | 
| 2026-06-07 14:28:38 | WS-CMP | 9567202 | cot-warm@7 seed36 (LSAT-100,ep7) + curve 9567203 | requeue | 
| 2026-06-07 18:57:22 | WS-SPICE | 9580756 | LAST: SPICE K=4 warm seed42 (LSAT, ep7, keep-all-ckpts) | `simpl_spice_oat.sh main/lsat-spice-warm-k4 42 qwen <base>` |
| 2026-06-07 18:57:22 | WS-SPICE | 9580758 | LAST: SPICE K=4 warm seed24 (LSAT, ep7, keep-all-ckpts) | `simpl_spice_oat.sh main/lsat-spice-warm-k4 24 qwen <base>` |
| 2026-06-07 18:57:22 | WS-SPICE | 9580760 | LAST: SPICE K=4 warm seed36 (LSAT, ep7, keep-all-ckpts) | `simpl_spice_oat.sh main/lsat-spice-warm-k4 36 qwen <base>` |
| 2026-06-07 22:11:45 | WS-2x2 | 9592319 | 2x2 lsat-2x2-cot-all seed42 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592322 | 2x2 lsat-2x2-cot-all seed24 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592324 | 2x2 lsat-2x2-cot-all seed36 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592327 | 2x2 lsat-2x2-simpl-all seed42 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592329 | 2x2 lsat-2x2-simpl-all seed24 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592332 | 2x2 lsat-2x2-simpl-all seed36 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592335 | 2x2 lsat-2x2-cot-front seed42 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592337 | 2x2 lsat-2x2-cot-front seed24 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592340 | 2x2 lsat-2x2-cot-front seed36 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592342 | 2x2 lsat-2x2-spice seed42 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592345 | 2x2 lsat-2x2-spice seed24 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:11:45 | WS-2x2 | 9592348 | 2x2 lsat-2x2-spice seed36 (warm, ep12, keep-all) + curve | requeue |
| 2026-06-07 22:44:41 | WS-2x2v2 | 9594604 | 2x2v2 lsat-2x2-cot-all seed42 (warm,ep12,K=1,keep-all) + curve 9594605 | `launch_2x2.sh` |
| 2026-06-07 22:44:41 | WS-2x2v2 | 9594606 | 2x2v2 lsat-2x2-cot-all seed24 (warm,ep12,K=1,keep-all) + curve 9594607 | `launch_2x2.sh` |
| 2026-06-07 22:44:41 | WS-2x2v2 | 9594608 | 2x2v2 lsat-2x2-cot-all seed36 (warm,ep12,K=1,keep-all) + curve 9594609 | `launch_2x2.sh` |
| 2026-06-07 22:44:41 | WS-2x2v2 | 9594610 | 2x2v2 lsat-2x2-simpl-all seed42 (warm,ep12,K=1,keep-all) + curve 9594611 | `launch_2x2.sh` |
| 2026-06-07 22:44:41 | WS-2x2v2 | 9594612 | 2x2v2 lsat-2x2-simpl-all seed24 (warm,ep12,K=1,keep-all) + curve 9594613 | `launch_2x2.sh` |
| 2026-06-07 22:44:42 | WS-2x2v2 | 9594614 | 2x2v2 lsat-2x2-simpl-all seed36 (warm,ep12,K=1,keep-all) + curve 9594615 | `launch_2x2.sh` |
| 2026-06-07 22:44:42 | WS-2x2v2 | 9594616 | 2x2v2 lsat-2x2-cot-front seed42 (warm,ep12,K=1,keep-all) + curve 9594617 | `launch_2x2.sh` |
| 2026-06-07 22:44:42 | WS-2x2v2 | 9594618 | 2x2v2 lsat-2x2-cot-front seed24 (warm,ep12,K=1,keep-all) + curve 9594619 | `launch_2x2.sh` |
| 2026-06-07 22:44:43 | WS-2x2v2 | 9594620 | 2x2v2 lsat-2x2-cot-front seed36 (warm,ep12,K=1,keep-all) + curve 9594621 | `launch_2x2.sh` |
| 2026-06-07 22:44:44 | WS-2x2v2 | 9594623 | 2x2v2 lsat-2x2-spice seed42 (warm,ep12,K=1,keep-all) + curve 9594624 | `launch_2x2.sh` |
| 2026-06-07 22:44:44 | WS-2x2v2 | 9594625 | 2x2v2 lsat-2x2-spice seed24 (warm,ep12,K=1,keep-all) + curve 9594626 | `launch_2x2.sh` |
| 2026-06-07 22:44:44 | WS-2x2v2 | 9594627 | 2x2v2 lsat-2x2-spice seed36 (warm,ep12,K=1,keep-all) + curve 9594628 | `launch_2x2.sh` |
| 2026-06-08 09:57:41 | WS-2x2v2 | 9626430 | 2x2v2 TRANSFER LSAT->RACE full-curve Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_42 | `fanout race-c` |
| 2026-06-08 09:57:42 | WS-2x2v2 | 9626431 | 2x2v2 TRANSFER LSAT->RACE full-curve Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_24 | `fanout race-c` |
| 2026-06-08 09:57:42 | WS-2x2v2 | 9626432 | 2x2v2 TRANSFER LSAT->RACE full-curve Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_36 | `fanout race-c` |
| 2026-06-08 09:57:42 | WS-2x2v2 | 9626433 | 2x2v2 TRANSFER LSAT->RACE full-curve Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_42 | `fanout race-c` |
| 2026-06-08 09:57:42 | WS-2x2v2 | 9626434 | 2x2v2 TRANSFER LSAT->RACE full-curve Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_24 | `fanout race-c` |
| 2026-06-08 09:57:42 | WS-2x2v2 | 9626435 | 2x2v2 TRANSFER LSAT->RACE full-curve Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_36 | `fanout race-c` |
| 2026-06-08 11:28:45 | WS-2x2v2 | 9633878 | 2x2v2 TRANSFER LSAT->RACE allsteps(inferno) Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_42 | `auto_curve_eval race-c` |
| 2026-06-08 11:28:46 | WS-2x2v2 | 9633879 | 2x2v2 TRANSFER LSAT->RACE allsteps(inferno) Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_24 | `auto_curve_eval race-c` |
| 2026-06-08 11:28:46 | WS-2x2v2 | 9633881 | 2x2v2 TRANSFER LSAT->RACE allsteps(inferno) Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_36 | `auto_curve_eval race-c` |
| 2026-06-08 11:28:46 | WS-2x2v2 | 9633882 | 2x2v2 TRANSFER LSAT->RACE allsteps(inferno) Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_42 | `auto_curve_eval race-c` |
| 2026-06-08 11:28:46 | WS-2x2v2 | 9633883 | 2x2v2 TRANSFER LSAT->RACE allsteps(inferno) Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_24 | `auto_curve_eval race-c` |
| 2026-06-08 11:28:46 | WS-2x2v2 | 9633885 | 2x2v2 TRANSFER LSAT->RACE allsteps(inferno) Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_36 | `auto_curve_eval race-c` |
| 2026-06-08 12:54:47 | WS-2x2v2 | 9642908 | 2x2v2 TRANSFER LSAT->RACE allsteps EMBERS-fixed(1024tok,incr) Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_42 | `auto_curve_eval race-c` |
| 2026-06-08 12:54:47 | WS-2x2v2 | 9642909 | 2x2v2 TRANSFER LSAT->RACE allsteps EMBERS-fixed(1024tok,incr) Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_24 | `auto_curve_eval race-c` |
| 2026-06-08 12:54:47 | WS-2x2v2 | 9642911 | 2x2v2 TRANSFER LSAT->RACE allsteps EMBERS-fixed(1024tok,incr) Qwen3-4B-Base-lsat-2x2-cot-all_simpl-oat_36 | `auto_curve_eval race-c` |
| 2026-06-08 12:54:48 | WS-2x2v2 | 9642912 | 2x2v2 TRANSFER LSAT->RACE allsteps EMBERS-fixed(1024tok,incr) Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_42 | `auto_curve_eval race-c` |
| 2026-06-08 12:54:48 | WS-2x2v2 | 9642914 | 2x2v2 TRANSFER LSAT->RACE allsteps EMBERS-fixed(1024tok,incr) Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_24 | `auto_curve_eval race-c` |
| 2026-06-08 12:54:48 | WS-2x2v2 | 9642915 | 2x2v2 TRANSFER LSAT->RACE allsteps EMBERS-fixed(1024tok,incr) Qwen3-4B-Base-lsat-2x2-simpl-all_simpl-oat_36 | `auto_curve_eval race-c` |
| 2026-06-08 14:29:37 | WS-PAIR | 9649763 | pair lsat-pair-simpl seed42 (base,ep2,pairs<=8,N) + curve 9649764 | `launch_pair.sh` |
| 2026-06-08 14:29:37 | WS-PAIR | 9649765 | pair lsat-pair-simpl seed24 (base,ep2,pairs<=8,N) + curve 9649766 | `launch_pair.sh` |
| 2026-06-08 14:29:37 | WS-PAIR | 9649767 | pair lsat-pair-simpl seed36 (base,ep2,pairs<=8,N) + curve 9649768 | `launch_pair.sh` |
| 2026-06-08 14:29:37 | WS-PAIR | 9649769 | pair lsat-pair-cot seed42 (base,ep2,pairs<=8,N) + curve 9649770 | `launch_pair.sh` |
| 2026-06-08 14:29:38 | WS-PAIR | 9649771 | pair lsat-pair-cot seed24 (base,ep2,pairs<=8,N) + curve 9649772 | `launch_pair.sh` |
| 2026-06-08 14:29:38 | WS-PAIR | 9649773 | pair lsat-pair-cot seed36 (base,ep2,pairs<=8,N) + curve 9649774 | `launch_pair.sh` |
| 2026-06-08 14:29:38 | WS-PAIR | 9649775 | pair lsat-pair-cot2x seed42 (base,ep2,pairs<=8,N) + curve 9649776 | `launch_pair.sh` |
| 2026-06-08 14:29:39 | WS-PAIR | 9649777 | pair lsat-pair-cot2x seed24 (base,ep2,pairs<=8,N) + curve 9649778 | `launch_pair.sh` |
| 2026-06-08 14:29:39 | WS-PAIR | 9649779 | pair lsat-pair-cot2x seed36 (base,ep2,pairs<=8,N) + curve 9649780 | `launch_pair.sh` |
| 2026-06-08 15:14:03 | WS-PAIR | 9653546 | pair-uall lsat-pair-simpl-uall seed42 (base,ep2,understand-all-q) + curve 9653547 | `launch` |
| 2026-06-08 15:14:03 | WS-PAIR | 9653549 | pair-uall lsat-pair-simpl-uall seed24 (base,ep2,understand-all-q) + curve 9653550 | `launch` |
| 2026-06-08 15:14:03 | WS-PAIR | 9653551 | pair-uall lsat-pair-simpl-uall seed36 (base,ep2,understand-all-q) + curve 9653552 | `launch` |
| 2026-06-08 22:34:52 | WS-DENSE | 9688583 | dense lsat-dense-simpl seed42 (warm,ep16,K=all,understand-all) + curve 9688584 | `launch_dense.sh` |
| 2026-06-08 22:34:55 | WS-DENSE | 9688585 | dense lsat-dense-simpl seed24 (warm,ep16,K=all,understand-all) + curve 9688586 | `launch_dense.sh` |
| 2026-06-08 22:34:55 | WS-DENSE | 9688587 | dense lsat-dense-simpl seed36 (warm,ep16,K=all,understand-all) + curve 9688588 | `launch_dense.sh` |
| 2026-06-08 22:34:55 | WS-DENSE | 9688589 | dense lsat-dense-cot seed42 (warm,ep16,K=all,understand-all) + curve 9688590 | `launch_dense.sh` |
| 2026-06-08 22:34:55 | WS-DENSE | 9688591 | dense lsat-dense-cot seed24 (warm,ep16,K=all,understand-all) + curve 9688592 | `launch_dense.sh` |
| 2026-06-08 22:34:56 | WS-DENSE | 9688593 | dense lsat-dense-cot seed36 (warm,ep16,K=all,understand-all) + curve 9688594 | `launch_dense.sh` |
| 2026-06-08 22:36:02 | WS-DENSE | 9688608 | dense lsat-dense-simpl seed42 (warm,ep16,K=all,understand-all) + curve 9688609 | `launch_dense.sh` |
| 2026-06-08 22:36:02 | WS-DENSE | 9688610 | dense lsat-dense-simpl seed24 (warm,ep16,K=all,understand-all) + curve 9688611 | `launch_dense.sh` |
| 2026-06-08 22:36:02 | WS-DENSE | 9688612 | dense lsat-dense-simpl seed36 (warm,ep16,K=all,understand-all) + curve 9688613 | `launch_dense.sh` |
| 2026-06-08 22:36:03 | WS-DENSE | 9688614 | dense lsat-dense-cot seed42 (warm,ep16,K=all,understand-all) + curve 9688615 | `launch_dense.sh` |
| 2026-06-08 22:36:03 | WS-DENSE | 9688616 | dense lsat-dense-cot seed24 (warm,ep16,K=all,understand-all) + curve 9688617 | `launch_dense.sh` |
| 2026-06-08 22:36:03 | WS-DENSE | 9688618 | dense lsat-dense-cot seed36 (warm,ep16,K=all,understand-all) + curve 9688619 | `launch_dense.sh` |
| 2026-06-08 22:53:03 | WS-PAIRWARM | 9689805 | pairwarm lsat-pair-uall-warm seed42 (warm,ep4,1:1 understand-all) + curve 9689806 | `launch_pairwarm.sh` |
| 2026-06-08 22:53:03 | WS-PAIRWARM | 9689807 | pairwarm lsat-pair-uall-warm seed24 (warm,ep4,1:1 understand-all) + curve 9689808 | `launch_pairwarm.sh` |
| 2026-06-08 22:53:04 | WS-PAIRWARM | 9689809 | pairwarm lsat-pair-uall-warm seed36 (warm,ep4,1:1 understand-all) + curve 9689810 | `launch_pairwarm.sh` |
| 2026-06-08 22:53:04 | WS-PAIRWARM | 9689811 | pairwarm lsat-pair-cot-warm seed42 (warm,ep4,1:1 understand-all) + curve 9689812 | `launch_pairwarm.sh` |
| 2026-06-08 22:53:04 | WS-PAIRWARM | 9689813 | pairwarm lsat-pair-cot-warm seed24 (warm,ep4,1:1 understand-all) + curve 9689814 | `launch_pairwarm.sh` |
| 2026-06-08 22:53:04 | WS-PAIRWARM | 9689815 | pairwarm lsat-pair-cot-warm seed36 (warm,ep4,1:1 understand-all) + curve 9689816 | `launch_pairwarm.sh` |
| 2026-06-09 10:39:09 | WS-DENSE | 9728852 | dense lsat-dense-simpl seed42 (warm,ep16,K=all,understand-all) + curve 9728854 | `launch_dense.sh` |
| 2026-06-09 10:39:10 | WS-DENSE | 9728855 | dense lsat-dense-simpl seed24 (warm,ep16,K=all,understand-all) + curve 9728856 | `launch_dense.sh` |
| 2026-06-09 10:39:10 | WS-DENSE | 9728858 | dense lsat-dense-simpl seed36 (warm,ep16,K=all,understand-all) + curve 9728859 | `launch_dense.sh` |
| 2026-06-09 10:39:10 | WS-DENSE | 9728860 | dense lsat-dense-cot seed42 (warm,ep16,K=all,understand-all) + curve 9728861 | `launch_dense.sh` |
| 2026-06-09 10:39:10 | WS-DENSE | 9728863 | dense lsat-dense-cot seed24 (warm,ep16,K=all,understand-all) + curve 9728864 | `launch_dense.sh` |
| 2026-06-09 10:39:10 | WS-DENSE | 9728865 | dense lsat-dense-cot seed36 (warm,ep16,K=all,understand-all) + curve 9728866 | `launch_dense.sh` |
| 2026-06-09 10:43:28 | WS-ROT24 | 9729262 | rotate understanding (simpl-all) seed42 ep24 warm + curve 9729264 | `launch` |
| 2026-06-09 10:43:29 | WS-ROT24 | 9729265 | rotate understanding (simpl-all) seed24 ep24 warm + curve 9729267 | `launch` |
| 2026-06-09 10:43:29 | WS-ROT24 | 9729268 | rotate understanding (simpl-all) seed36 ep24 warm + curve 9729269 | `launch` |
| 2026-06-09 19:50:00 | WS-XFER | 9764390 | TRANSFER LSAT->RACE und lsat-2x2-simpl-all seed42 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:00 | WS-XFER | 9764391 | TRANSFER LSAT->RACE und lsat-2x2-simpl-all seed24 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:03 | WS-XFER | 9764392 | TRANSFER LSAT->RACE und lsat-2x2-simpl-all seed36 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:03 | WS-XFER | 9764393 | TRANSFER LSAT->RACE cot lsat-pair-cot-warm seed42 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:03 | WS-XFER | 9764394 | TRANSFER LSAT->RACE cot lsat-pair-cot-warm seed24 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:04 | WS-XFER | 9764395 | TRANSFER LSAT->RACE cot lsat-pair-cot-warm seed36 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:57 | WS-XFER | 9764409 | TRANSFER LSAT->PWd5 und lsat-2x2-simpl-all seed42 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:57 | WS-XFER | 9764410 | TRANSFER LSAT->PWd5 und lsat-2x2-simpl-all seed24 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:57 | WS-XFER | 9764411 | TRANSFER LSAT->PWd5 und lsat-2x2-simpl-all seed36 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:58 | WS-XFER | 9764412 | TRANSFER LSAT->PWd5 cot lsat-pair-cot-warm seed42 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:58 | WS-XFER | 9764414 | TRANSFER LSAT->PWd5 cot lsat-pair-cot-warm seed24 (rotate24 study) | `auto_curve_eval` |
| 2026-06-09 19:50:58 | WS-XFER | 9764415 | TRANSFER LSAT->PWd5 cot lsat-pair-cot-warm seed36 (rotate24 study) | `auto_curve_eval` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792602 | DOWNLOAD Qwen3-32B-Base + OctoThinker-{3B,8B}-Hybrid-Base | `download_models` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792603 | baseline-v2 qwen3-4b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792604 | baseline-v2 qwen3-4b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792605 | baseline-v2 qwen3-4b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792606 | baseline-v2 qwen3-8b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792607 | baseline-v2 qwen3-8b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792608 | baseline-v2 qwen3-8b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:07 | WS-BASELINE | 9792609 | baseline-v2 qwen3-32b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792610 | baseline-v2 qwen3-32b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792611 | baseline-v2 qwen3-32b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792612 | baseline-v2 octothinker-3b-hybrid-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792613 | baseline-v2 octothinker-3b-hybrid-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792614 | baseline-v2 octothinker-3b-hybrid-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792615 | baseline-v2 octothinker-8b-hybrid-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792616 | baseline-v2 octothinker-8b-hybrid-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 11:32:08 | WS-BASELINE | 9792617 | baseline-v2 octothinker-8b-hybrid-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801876 | baseline-v2 qwen3-4b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801877 | baseline-v2 qwen3-4b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801878 | baseline-v2 qwen3-4b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801879 | baseline-v2 qwen3-8b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801880 | baseline-v2 qwen3-8b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801881 | baseline-v2 qwen3-8b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801882 | baseline-v2 qwen3-32b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:32 | WS-BASELINE | 9801883 | baseline-v2 qwen3-32b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801884 | baseline-v2 qwen3-32b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801885 | baseline-v2 octothinker-3b-hybrid-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801886 | baseline-v2 octothinker-3b-hybrid-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801887 | baseline-v2 octothinker-3b-hybrid-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801888 | baseline-v2 octothinker-8b-hybrid-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801889 | baseline-v2 octothinker-8b-hybrid-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 18:41:33 | WS-BASELINE | 9801890 | baseline-v2 octothinker-8b-hybrid-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803691 | baseline-v2 qwen3-4b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803692 | baseline-v2 qwen3-4b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803694 | baseline-v2 qwen3-4b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803695 | baseline-v2 qwen3-8b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803696 | baseline-v2 qwen3-8b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803698 | baseline-v2 qwen3-8b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803699 | baseline-v2 qwen3-32b-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:10 | WS-BASELINE | 9803700 | baseline-v2 qwen3-32b-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803702 | baseline-v2 qwen3-32b-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803704 | baseline-v2 octothinker-3b-hybrid-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803706 | baseline-v2 octothinker-3b-hybrid-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803707 | baseline-v2 octothinker-3b-hybrid-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803708 | baseline-v2 octothinker-8b-hybrid-base x lsat avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803709 | baseline-v2 octothinker-8b-hybrid-base x race avg@8 | `run_eval_untrained` |
| 2026-06-10 19:23:13 | WS-BASELINE | 9803710 | baseline-v2 octothinker-8b-hybrid-base x pwd5 avg@8 | `run_eval_untrained` |
| 2026-06-10 23:11:30 | WS-WARMUP | 9815628 | cot warm-up HP lr4e6 (LSAT-100, ep3, from base) | `cot_oat` |
| 2026-06-10 23:11:30 | WS-WARMUP | 9815629 | cot warm-up HP lr8e6 (LSAT-100, ep3, from base) | `cot_oat` |
| 2026-06-10 23:11:30 | WS-WARMUP | 9815630 | cot warm-up HP lr16e6 (LSAT-100, ep3, from base) | `cot_oat` |
| 2026-06-11 13:19:32 | WS-WARMUP | 9832227 | cot warm-up HP lr32e6 (LSAT-100,ep3) + curve 9832228 | `cot_oat` |
| 2026-06-11 13:19:32 | WS-WARMUP | 9832230 | cot warm-up HP lr64e6 (LSAT-100,ep3) + curve 9832231 | `cot_oat` |
| 2026-06-11 13:24:01 | WS-SIMPLBASE | 9832448 | simpl from-base lr16e6 (LSAT-100,ep18=216steps) + curve 9832449 | `simpl_oat` |
| 2026-06-11 13:24:01 | WS-SIMPLBASE | 9832450 | simpl from-base lr32e6 (LSAT-100,ep18=216steps) + curve 9832451 | `simpl_oat` |
| 2026-06-11 19:30:09 | WS-WARMUP | 9845708 | cot warm-up lr32e6 seed24 (LSAT-100,ep3) + curve 9845709 | `cot_oat` |
| 2026-06-11 19:30:10 | WS-WARMUP | 9845710 | cot warm-up lr32e6 seed36 (LSAT-100,ep3) + curve 9845711 | `cot_oat` |
| 2026-06-11 19:31:26 | WS-SIMPLBASE | 9845739 | simpl from-base lr64e6 (LSAT-100,ep18=216steps) + curve 9845740 | `simpl_oat` |
| 2026-06-11 22:54:30 | WS-SIMPLBASE | 9852392 | simpl from-base lr32e6 seed24 (LSAT-100,ep18) + curve 9852394 | `simpl_oat` |
| 2026-06-11 22:54:30 | WS-SIMPLBASE | 9852395 | simpl from-base lr32e6 seed36 (LSAT-100,ep18) + curve 9852396 | `simpl_oat` |
| 2026-06-11 23:21:49 | WS-RACE | 9853165/9853167 | race cot(ep3)+simpl(ep17) lr16e6 from base seed42 | `launch` |
| 2026-06-11 23:21:49 | WS-RACE | 9853169/9853171 | race cot(ep3)+simpl(ep17) lr32e6 from base seed42 | `launch` |
| 2026-06-11 23:21:49 | WS-RACE | 9853173/9853175 | race cot(ep3)+simpl(ep17) lr64e6 from base seed42 | `launch` |
| 2026-06-12 09:18:17 | WS-RACE | 9862839/9862841 | race 2x cot(ep6)+simpl(ep34) lr64e6 from base | `launch` |
| 2026-06-12 09:18:17 | WS-RACE | 9862843/9862845 | race 2x cot(ep6)+simpl(ep34) lr128e6 from base | `launch` |
| 2026-06-12 09:19:33 | WS-LSATSIZE | 9862855/9862857 | lsat50 cot+simpl lr32e6 seed142 | `launch` |
| 2026-06-12 09:19:33 | WS-LSATSIZE | 9862859/9862861 | lsat200 cot+simpl lr32e6 seed142 | `launch` |
| 2026-06-13 10:35:30 | WS-RACEPROMPT | 9898602 | race simpl prompt-v2 lr32e6 seed42 (stronger understanding prompt) + curve 9898603 | `simpl_oat` |
| 2026-06-13 15:59:49 | WS-CTRL | 9911706 | cot N=16 (compute-match) LSAT-100 ep5 seed42 | `cot_oat` |
| 2026-06-13 15:59:50 | WS-CTRL | 9911708 | cot N=16 (compute-match) LSAT-100 ep5 seed24 | `cot_oat` |
| 2026-06-13 15:59:51 | WS-CTRL | 9911710 | cot N=16 (compute-match) LSAT-100 ep5 seed36 | `cot_oat` |
| 2026-06-14 10:44:50 | WS-FINAL | final-lsat50-cot | final lsat50_cot 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
| 2026-06-14 10:44:51 | WS-FINAL | final-lsat50-cotn16 | final lsat50_cotn16 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
| 2026-06-14 10:44:52 | WS-FINAL | final-lsat50-simpl | final lsat50_simpl 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
| 2026-06-14 10:44:53 | WS-FINAL | final-lsat100-cot | final lsat100_cot 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
| 2026-06-14 10:44:55 | WS-FINAL | final-lsat100-simpl | final lsat100_simpl 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
| 2026-06-14 10:44:56 | WS-FINAL | final-race100-cot | final race100_cot 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
| 2026-06-14 10:44:57 | WS-FINAL | final-race100-simpl | final race100_simpl 3 seeds (lr3.2e5/pe4/rs1) | `launch` |
