#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Configuration: edit these
# ------------------------
PYTHON_CMD="python3"   # or path to your python in conda env: /path/to/python
SCRIPT="final_final_gnn.py"  # python training script filename
HALLU_DIRS=( "hallucinating_circuits/code_dataset" "hallucinating_circuits/medal_1500_mc" "hallucinating_circuits/medical_abstract_1500" "hallucinating_circuits/numerical_dataset" "hallucinating_circuits/prompts_rationalization_binary")
NONHALLU_DIRS=( "Non_hallucinating_circuits/code_dataset" "Non_hallucinating_circuits/medal_1500_mc" "Non_hallucinating_circuits/medical_abstract_1500" "Non_hallucinating_circuits/numerical_dataset" "Non_hallucinating_circuits/prompts_rationalization_binary" )
NODE_FEATURES_DIR="./"   # directory where circuit_X_node_*.csv live (set to actual)
RESULTS_CSV="results_summary.csv"

# create/clear results csv header (optional)
# Uncomment the following line if you want to overwrite existing results_summary.csv
# echo "run_name,use_gat,gat_heads,hidden_channels,top_k_csv,selected_csv_cols,use_name_feats,epochs,final_val_acc,run_dir" > $RESULTS_CSV

# ------------------------
# Hyperparameter grid
# ------------------------
TOP_K_LIST=(0 5 10)                # 0 = don't auto-select; use selected_csv_cols instead
HIDDEN_LIST=(32 64)
USE_GAT_LIST=(false true)
GAT_HEADS_LIST=(2 4)
LR_LIST=(0.001 0.0005)
BATCH_LIST=(16 32)
EPOCHS=30

# Optional: provide explicit selected CSV columns (overrides top_k)
# Leave empty to use auto selection (if top_k > 0), or to use default union of CSV numeric columns.
SELECTED_CSV_COLS=""  # example: "colA,colB"

# ------------------------
# Loop through grid
# ------------------------
run_counter=0
for topk in "${TOP_K_LIST[@]}"; do
  for hidden in "${HIDDEN_LIST[@]}"; do
    for use_gat in "${USE_GAT_LIST[@]}"; do
      for heads in "${GAT_HEADS_LIST[@]}"; do
        for lr in "${LR_LIST[@]}"; do
          for batch in "${BATCH_LIST[@]}"; do

            run_counter=$((run_counter+1))
            timestamp=$(date +"%Y%m%d-%H%M%S")
            run_name="run_${timestamp}_r${run_counter}_top${topk}_gat${use_gat}_h${hidden}_gh${heads}_lr${lr}_bs${batch}"
            run_dir="runs/${run_name}"
            mkdir -p "${run_dir}"
            log_file="${run_dir}/run.log"

            echo "================================================================"
            echo "Starting ${run_name}"
            echo "run_dir = ${run_dir}"
            echo "Logging to ${log_file}"
            echo "================================================================"

            # Build argument list
            ARGS=( 
              --hallu_folders "${HALLU_DIRS[@]}" 
              --nonhallu_folders "${NONHALLU_DIRS[@]}"
              --node_features_dir "${NODE_FEATURES_DIR}"
              --hidden_channels "${hidden}"
              --gat_heads "${heads}"
              --batch_size "${batch}"
              --lr "${lr}"
              --epochs "${EPOCHS}"
              --run_dir "${run_dir}"
              --results_csv "${RESULTS_CSV}"
            )

            # boolean flag for use_gat
            if [ "${use_gat}" = "true" ] ; then
              ARGS+=( --use_gat )
            fi

            # top_k or explicit selected CSVs
            ARGS+=( --top_k_csv "${topk}" )
            if [ -n "${SELECTED_CSV_COLS}" ]; then
              ARGS+=( --selected_csv_cols "${SELECTED_CSV_COLS}" )
            fi

            # Launch the Python script, tee output to log file
            (
              echo "Command: ${PYTHON_CMD} ${SCRIPT} ${ARGS[*]}"
              ${PYTHON_CMD} "${SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${log_file}"
            )

            echo "Finished ${run_name} (logs -> ${log_file})"
            sleep 1  # small pause to avoid hammering disk
          done
        done
      done
    done
  done
done

echo "All runs completed. Total runs: ${run_counter}"
echo "Results appended to ${RESULTS_CSV}"
