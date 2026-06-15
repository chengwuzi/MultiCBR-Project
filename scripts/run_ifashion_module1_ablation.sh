#!/usr/bin/env bash
set -uo pipefail

GPU=${1:-0}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
OUT_DIR="outputs/module1_ablation"
LOG_DIR="${OUT_DIR}/logs"
STATUS_FILE="${OUT_DIR}/experiment_status.json"

mkdir -p "${LOG_DIR}"

touch_status_file() {
  python -c "import json,pathlib; p=pathlib.Path('${STATUS_FILE}'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}', encoding='utf-8') if not p.exists() else None"
}

is_done() {
  local exp_id="$1"
  python -c "import json,sys,pathlib; p=pathlib.Path('${STATUS_FILE}'); data=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; sys.exit(0 if data.get('${exp_id}', {}).get('status') == 'success' else 1)"
}

set_status() {
  local exp_id="$1"
  local group="$2"
  local strategy="$3"
  local k="$4"
  local status="$5"
  local attempts="$6"
  local log_path="$7"
  python -c "import json,pathlib,time; p=pathlib.Path('${STATUS_FILE}'); data=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; data['${exp_id}']={'method_group':'${group}','strategy':'${strategy}','k':'${k}','status':'${status}','attempts':int('${attempts}'),'log_path':'${log_path}','updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}; p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')"
}

has_result_metrics() {
  local log_path="$1"
  python -c "import re,sys,pathlib; p=pathlib.Path('${log_path}'); text=p.read_text(encoding='utf-8', errors='ignore') if p.exists() else ''; ok=all(re.search(r'Best in epoch\\s+\\d+,\\s+TOP\\s+%d:\\s+REC_T=' % k, text) for k in (10,20,40,80)); sys.exit(0 if ok else 1)"
}

run_experiment() {
  local exp_id="$1"
  local group="$2"
  local strategy="$3"
  local k="$4"

  if is_done "${exp_id}"; then
    echo "[SKIP] ${exp_id} already completed."
    return 0
  fi

  local attempt=1
  while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local log_path="${LOG_DIR}/${exp_id}_attempt${attempt}_${ts}.log"
    echo "================================================================"
    echo "[RUN] experiment=${exp_id} group=${group} strategy=${strategy} k=${k} attempt=${attempt}/${MAX_ATTEMPTS} gpu=${GPU}"
    echo "================================================================"

    if [ "${strategy}" = "raw" ]; then
      python train.py -d iFashion -g "${GPU}" -i "${exp_id}" --dwt_rebuild_strategy raw 2>&1 | tee "${log_path}"
    else
      python train.py -d iFashion -g "${GPU}" -i "${exp_id}" --dwt_rebuild_strategy "${strategy}" --dwt_rebuild_k "${k}" 2>&1 | tee "${log_path}"
    fi
    local exit_code=${PIPESTATUS[0]}

    if [ "${exit_code}" -eq 0 ] && has_result_metrics "${log_path}"; then
      echo "[OK] ${exp_id} completed on attempt ${attempt}."
      set_status "${exp_id}" "${group}" "${strategy}" "${k}" "success" "${attempt}" "${log_path}"
      python scripts/collect_ifashion_module1_results.py
      return 0
    fi

    if [ "${exit_code}" -eq 0 ]; then
      echo "[WARN] ${exp_id} exited successfully but best test metrics were not found. See ${log_path}"
    else
      echo "[WARN] ${exp_id} failed on attempt ${attempt} with exit_code=${exit_code}. See ${log_path}"
    fi
    set_status "${exp_id}" "${group}" "${strategy}" "${k}" "retrying" "${attempt}" "${log_path}"
    attempt=$((attempt + 1))
    sleep 10
  done

  echo "[FAILED] ${exp_id} failed after ${MAX_ATTEMPTS} attempts; skipping."
  set_status "${exp_id}" "${group}" "${strategy}" "${k}" "failed_skipped" "${MAX_ATTEMPTS}" "${log_path}"
  python scripts/collect_ifashion_module1_results.py
  return 0
}

touch_status_file

run_experiment "M1Raw" "Raw" "raw" "all"

for k in 1 3 5 7 9; do
  run_experiment "M1A_diff_k${k}" "A" "diffusion" "${k}"
done

for k in 1 3 5 7 9; do
  run_experiment "M1B_rand_k${k}" "B" "random" "${k}"
done

for k in 1 3 5 7 9; do
  run_experiment "M1C_pre_k${k}" "C" "pretrain_sim" "${k}"
done

for k in 1 3 5 7 9; do
  run_experiment "M1D_pop_k${k}" "D" "popularity" "${k}"
done

for k in 1 3 5 7 9; do
  run_experiment "M1E_z0_k${k}" "E" "pooled_z0" "${k}"
done

python scripts/collect_ifashion_module1_results.py
echo "[DONE] Module-one iFashion ablation queue finished. Summary: ${OUT_DIR}/summary.txt"
