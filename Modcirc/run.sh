#!/usr/bin/env bash
set -u
# Do not set -e so we continue on errors for other files; we log failures per-file.
# If you prefer to stop on first error, uncomment: set -e

source ~/data1/miniconda3/etc/profile.d/conda.sh
conda activate modcirc_env
source ~/.bashrc



ROOT_DIR="$(pwd)"
FINAL_DATASET="${ROOT_DIR}/FinalDataset"
TEXT_DEST="${ROOT_DIR}/text_dataset/hallucination_data"
NUMERIC_JSONL="${TEXT_DEST}/numerical_dataset.jsonl"
RESULTS_DIR="${ROOT_DIR}/results"
BATCH_SCRIPT="${ROOT_DIR}/batch_extract_circuit_features.py"
EXTRACTOR_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LOGFILE="${ROOT_DIR}/batch_run.log"
#LOGFILE="/dev/null"


GRAPHS_DIR="${ROOT_DIR}/graphs"
HALLUC_FOLDER="${GRAPHS_DIR}/hallucinating_circuits"
NONHALLUC_FOLDER="${GRAPHS_DIR}/Non_hallucinating_circuits"

BATCH_CREATOR="${ROOT_DIR}/create_batches.py"
DEFAULT_PCT=100         # % of dataset per batch (adjust as needed)
DEFAULT_MIN_BATCHES=1
SEED=42                # optional deterministic seed; set to "" to randomize

# Ensure log file exists (append mode)
echo "=== Batch run started at $(date) ===" >> "${LOGFILE}"

mkdir -p "${HALLUC_FOLDER}" "${NONHALLUC_FOLDER}"
mkdir -p "${TEXT_DEST}"

# helper to reset results folder structure
reset_results_dirs() {
  echo "[info] Clearing ${RESULTS_DIR} and recreating required subfolders..." | tee -a "${LOGFILE}"
  if [ -d "${RESULTS_DIR}" ]; then
    rm -rf "${RESULTS_DIR:?}"/*
  else
    mkdir -p "${RESULTS_DIR}"
  fi
  mkdir -p "${RESULTS_DIR}/after_cutting_new" "${RESULTS_DIR}/before_cutting_new" \
           "${RESULTS_DIR}/circuits_after_new" "${RESULTS_DIR}/circuits_before_new"
}

# initial ensure results structure is present
reset_results_dirs

# process a directory of jsonl files and place outputs in target graph folder
process_jsonl_dir() {
  local src_dir="$1"
  local dest_graph_parent="$2"   # either HALLUC_FOLDER or NONHALLUC_FOLDER

  if [ ! -d "${src_dir}" ]; then
    echo "[warn] source directory does not exist: ${src_dir}" | tee -a "${LOGFILE}"
    return 0
  fi

  # iterate jsonl files (non-recursively)
  for jsonlfile in "${src_dir}"/*.jsonl; do
    # handle case when glob doesn't match (shell leaves literal)
    if [ ! -e "${jsonlfile}" ]; then
      echo "[info] no jsonl files found in ${src_dir}" | tee -a "${LOGFILE}"
      break
    fi

    jsonlbase="$(basename "${jsonlfile}" .jsonl)"
    echo "-----" | tee -a "${LOGFILE}"
    echo "[start] Processing ${jsonlfile} (name: ${jsonlbase}) at $(date)" | tee -a "${LOGFILE}"

    # create batches for this jsonl
    batch_outdir="${TEXT_DEST}/batches/${jsonlbase}"
    mkdir -p "${batch_outdir}"
    echo "[info] Creating batches for ${jsonlfile} -> ${batch_outdir}" | tee -a "${LOGFILE}"

    if [ -n "${SEED}" ]; then
      python3 "${BATCH_CREATOR}" --input "${jsonlfile}" --outdir "${batch_outdir}" --pct "${DEFAULT_PCT}" --min-batches "${DEFAULT_MIN_BATCHES}" --seed "${SEED}" >> "${LOGFILE}" 2>&1
    else
      python3 "${BATCH_CREATOR}" --input "${jsonlfile}" --outdir "${batch_outdir}" --pct "${DEFAULT_PCT}" --min-batches "${DEFAULT_MIN_BATCHES}" >> "${LOGFILE}" 2>&1
    fi

    if [ $? -ne 0 ]; then
      echo "[error] create_batches.py failed for ${jsonlbase}" | tee -a "${LOGFILE}"
      # continue to next file
      continue
    fi

    # iterate over created batches
    for batchfile in "${batch_outdir}"/*.jsonl; do
      if [ ! -e "${batchfile}" ]; then
        echo "[warn] no batch files found in ${batch_outdir}" | tee -a "${LOGFILE}"
        break
      fi

      batchbase="$(basename "${batchfile}" .jsonl)"
      echo "[start-batch] Processing batch ${batchbase} from ${jsonlbase} at $(date)" | tee -a "${LOGFILE}"

      # copy this batch into the numeric JSONL path expected by main.py
      cp -f "${batchfile}" "${NUMERIC_JSONL}"
      if [ $? -ne 0 ]; then
        echo "[error] failed to copy ${batchfile} -> ${NUMERIC_JSONL}" | tee -a "${LOGFILE}"
        continue
      fi
      echo "[ok] copied batch -> ${NUMERIC_JSONL}" | tee -a "${LOGFILE}"
      echo "reconneting wifi :( "
      python3 ~/misc/internet/proxyiit.py 
      # run main.py (assumes correct conda env activated)
      echo "[run] python3 main.py (jsonl: ${batchbase})" | tee -a "${LOGFILE}"
      python3 "${ROOT_DIR}/main.py" --model llama2 --num_exp 1 --exp_type kmeans --epochs 1 --save_path "${RESULTS_DIR}" >> "${LOGFILE}" 2>&1
      MAIN_EXIT=$?
      if [ ${MAIN_EXIT} -ne 0 ]; then
        echo "[error] main.py failed for ${batchbase} (exit ${MAIN_EXIT}). See ${LOGFILE} for details." | tee -a "${LOGFILE}"
        # proceed to attempt extractor anyway
      else
        echo "[ok] main.py completed for ${batchbase}" | tee -a "${LOGFILE}"
      fi


      # run the batch extractor
      echo "reconneting wifi :( "
      python3 ~/misc/internet/proxyiit.py 
      echo "[run] extractor for circuits in results/circuits_before_new" | tee -a "${LOGFILE}"
      python3 "${BATCH_SCRIPT}" \
        --model_name_or_path "${EXTRACTOR_MODEL}" \
        --circuits_dir "${RESULTS_DIR}/circuits_before_new" \
        --out_dir "${RESULTS_DIR}/circuits_before_new" \
        --pattern "circuit_*.pkl" >> "${LOGFILE}" 2>&1
      EXTR_EXIT=$?
      if [ ${EXTR_EXIT} -ne 0 ]; then
        echo "[error] batch_extract_circuit_features failed for ${jsonlbase} (exit ${EXTR_EXIT}). See ${LOGFILE}." | tee -a "${LOGFILE}"
        # still attempt to copy whatever is present
      else
        echo "[ok] extractor finished for ${jsonlbase}" | tee -a "${LOGFILE}"
      fi

      # # run dynamic aggregator (same as before)
      # echo "[run] dynamic aggregator for circuits in results/circuits_before_new" | tee -a "${LOGFILE}"
      # python3 "${ROOT_DIR}/batch_extract_dynamic_aggregate.py" \
      #   --model_name_or_path "${EXTRACTOR_MODEL}" \
      #   --circuits_dir "${RESULTS_DIR}/circuits_before_new" \
      #   --out_dir "${RESULTS_DIR}/circuits_before_new" \
      #   --pattern "circuit_*.pkl" \
      #   --prompts_file "${NUMERIC_JSONL}" \
      #   --save_raw_npz \
      #   --overwrite >> "${LOGFILE}" 2>&1
      # EXTR_EXIT=$?
      # if [ ${EXTR_EXIT} -ne 0 ]; then
      #   echo "[error] batch_extract_dynamic_aggregate failed for ${batchbase} (exit ${EXTR_EXIT}). See ${LOGFILE}." | tee -a "${LOGFILE}"
      # else
      #   echo "[ok] dynamic aggregator finished for ${batchbase}" | tee -a "${LOGFILE}"
      # fi

      # Prepare destination folder for this batch under the correct graph category
      dest_folder="${dest_graph_parent}/${jsonlbase}/${batchbase}"
      mkdir -p "${dest_folder}"

      # Copy the contents of ./results/circuits_before_new into destination
      if [ -d "${RESULTS_DIR}/circuits_before_new" ]; then
        echo "[copy] Copying ${RESULTS_DIR}/circuits_before_new -> ${dest_folder}" | tee -a "${LOGFILE}"
        rsync -a --delete "${RESULTS_DIR}/circuits_before_new/" "${dest_folder}/" >> "${LOGFILE}" 2>&1
        if [ $? -eq 0 ]; then
          echo "[ok] copied circuits_before_new to ${dest_folder}" | tee -a "${LOGFILE}"
        else
          echo "[warn] rsync copy may have failed for ${batchbase}" | tee -a "${LOGFILE}"
        fi
      else
        echo "[warn] ${RESULTS_DIR}/circuits_before_new does not exist for ${batchbase}" | tee -a "${LOGFILE}"
      fi

      # AFTER copying, reset results folder structure before next iteration
      reset_results_dirs

      echo "[done-batch] finished processing ${batchfile} at $(date)" | tee -a "${LOGFILE}"
    done

    echo "[done] finished processing ${jsonlfile} at $(date)" | tee -a "${LOGFILE}"
  done
}

# process hallucination folder -> hallucinating_circuits
process_jsonl_dir "${FINAL_DATASET}/hallucination_HaluEval" "${HALLUC_FOLDER}"

# process non-hallucination folder -> Non_hallucinating_circuits
process_jsonl_dir "${FINAL_DATASET}/Non_hallucination_HaluEval" "${NONHALLUC_FOLDER}"

echo "=== Batch run finished at $(date) ===" >> "${LOGFILE}"
echo "Done. See ${LOGFILE} for details."
