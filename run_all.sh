#!/usr/bin/env bash
# Run the full IDH pipeline for all datasets and models.
#
# Usage:
#   ./run_all.sh                          # run everything (except skull-strip)
#   SKIP_SKULL_STRIP=0 ./run_all.sh       # HD-BET on ERASMUS-GBM (first run)
#   SKIP_EMBEDDINGS=1  ./run_all.sh
#   SKIP_RADIOMICS=1   ./run_all.sh
#   SKIP_TABPFN=1      ./run_all.sh
#   SKIP_UCSD=1        ./run_all.sh       # skip UCSD-PTGBM external test

set -euo pipefail

cd "$(dirname "$0")"

DATASETS=(UCSF-PDGM UPENN-GBM ERASMUS-GBM UTSW-Glioma)
MODELS=(brainiac mricore biomedclip braindino)

UCSD_AVAILABLE=0
if [[ "${SKIP_UCSD:-0}" != "1" && -d "data/UCSD-PTGBM" ]]; then
  UCSD_AVAILABLE=1
fi

declare -A T1_SEQ=(
  [UCSF-PDGM]=T1c
  [UPENN-GBM]=T1GD
  [ERASMUS-GBM]=T1GD
  [UTSW-Glioma]=T1c
  [UCSD-PTGBM]=T1c
)

declare -A IDH_EXTERNAL=(
  [UCSF-PDGM]="UPENN-GBM ERASMUS-GBM UTSW-Glioma UCSD-PTGBM"
  [UPENN-GBM]="UCSF-PDGM ERASMUS-GBM UTSW-Glioma UCSD-PTGBM"
  [ERASMUS-GBM]="UCSF-PDGM UPENN-GBM UTSW-Glioma UCSD-PTGBM"
  [UTSW-Glioma]="UCSF-PDGM UPENN-GBM ERASMUS-GBM UCSD-PTGBM"
)

log() { printf '\n========== %s ==========\n\n' "$*"; }

filter_externals() {
  local filtered=()
  for e in $1; do
    if [[ "$e" == "UCSD-PTGBM" && "$UCSD_AVAILABLE" != "1" ]]; then
      continue
    fi
    filtered+=("$e")
  done
  echo "${filtered[@]}"
}

# Pre-flight: fail fast if TabPFN will run without a token.
if [[ "${SKIP_RADIOMICS:-0}" != "1" && "${SKIP_TABPFN:-0}" != "1" && -z "${TABPFN_TOKEN:-}" ]]; then
  echo "[ERROR] TABPFN_TOKEN is not set but the TabPFN stage is enabled."
  echo "        export TABPFN_TOKEN=\"...\"     # see README"
  echo "        SKIP_TABPFN=1 ./run_all.sh     # skip just TabPFN"
  echo "        SKIP_RADIOMICS=1 ./run_all.sh  # skip all radiomics stages"
  exit 1
fi

# 1. Skull-strip ERASMUS-GBM (one-time after download; idempotent).
if [[ "${SKIP_SKULL_STRIP:-1}" != "1" ]]; then
  log "Skull-stripping ERASMUS-GBM"
  python src/preprocessing/skull_strip.py --dataset ERASMUS-GBM --sequence FLAIR
  python src/preprocessing/skull_strip.py --dataset ERASMUS-GBM --sequence T1GD
fi

# 2. MAIN csvs (must run before embeddings: BrainIAC's extractor reads labels)
log "UPENN csv assembly (IDH)"
python src/preprocessing/upenn_csv_assembly.py

log "ERASMUS csv assembly (IDH)"
python src/preprocessing/erasmus_csv_assembly.py

log "UCSF csv assembly (IDH)"
python src/preprocessing/ucsf_csv_assembly.py

log "UTSW csv assembly (IDH)"
python src/preprocessing/utsw_csv_assembly.py

if [[ "$UCSD_AVAILABLE" == "1" ]]; then
  log "UCSD-PTGBM csv assembly (IDH)"
  python src/preprocessing/ucsd_csv_assembly.py
fi

# 3. Splits
for dataset in "${DATASETS[@]}"; do
  log "Splits: $dataset (IDH)"
  python src/preprocessing/generate_splits.py --dataset "$dataset"
done

# 4. Embeddings
if [[ "${SKIP_EMBEDDINGS:-0}" != "1" ]]; then
  EMB_DATASETS=("${DATASETS[@]}")
  if [[ "$UCSD_AVAILABLE" == "1" ]]; then
    EMB_DATASETS+=(UCSD-PTGBM)
  fi
  for model in "${MODELS[@]}"; do
    for dataset in "${EMB_DATASETS[@]}"; do
      log "Embeddings: $model / $dataset / FLAIR"
      python "src/embeddings/get_emb_${model}.py" --dataset "$dataset" --sequence FLAIR

      log "Embeddings: $model / $dataset / ${T1_SEQ[$dataset]}"
      python "src/embeddings/get_emb_${model}.py" --dataset "$dataset" --sequence "${T1_SEQ[$dataset]}"
    done
  done
fi

# 5. Linear probe
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    ext="$(filter_externals "${IDH_EXTERNAL[$dataset]:-}")"
    if [[ -n "$ext" ]]; then
      log "Linear probe: $model / $dataset (IDH) → external $ext"
      # shellcheck disable=SC2086
      python src/training/train_lin_probe.py --dataset "$dataset" --model "$model" \
        --external-test $ext
    else
      log "Linear probe: $model / $dataset (IDH)"
      python src/training/train_lin_probe.py --dataset "$dataset" --model "$model"
    fi
  done
done

# 6. Radiomics pipeline
if [[ "${SKIP_RADIOMICS:-0}" != "1" ]]; then

  RADIOMICS_DATASETS=("${DATASETS[@]}")
  if [[ "$UCSD_AVAILABLE" == "1" ]]; then
    RADIOMICS_DATASETS+=(UCSD-PTGBM)
  fi

  for dataset in "${RADIOMICS_DATASETS[@]}"; do
    log "Radiomics preprocessing: $dataset"
    python src/preprocessing/rfe_preprocessing.py --dataset "$dataset"
  done

  for dataset in "${RADIOMICS_DATASETS[@]}"; do
    log "Radiomics feature extraction: $dataset"
    python src/preprocessing/extract_radiomics.py --dataset "$dataset"
  done

  for dataset in "${DATASETS[@]}"; do
    log "Radiomics merge: $dataset (IDH)"
    python src/preprocessing/clean_radiomics.py --dataset "$dataset"
  done
  if [[ "$UCSD_AVAILABLE" == "1" ]]; then
    log "Radiomics merge: UCSD-PTGBM (IDH)"
    python src/preprocessing/clean_radiomics.py --dataset UCSD-PTGBM
  fi

  for dataset in "${DATASETS[@]}"; do
    ext="$(filter_externals "${IDH_EXTERNAL[$dataset]:-}")"
    if [[ -n "$ext" ]]; then
      log "LogReg: $dataset (IDH) → external $ext"
      # shellcheck disable=SC2086
      python src/training/train_log_reg.py --dataset "$dataset" --external-test $ext
    else
      log "LogReg: $dataset (IDH)"
      python src/training/train_log_reg.py --dataset "$dataset"
    fi
  done

  if [[ "${SKIP_TABPFN:-0}" != "1" ]]; then
    for dataset in "${DATASETS[@]}"; do
      ext="$(filter_externals "${IDH_EXTERNAL[$dataset]:-}")"
      if [[ -n "$ext" ]]; then
        log "TabPFN: $dataset (IDH) → external $ext"
        # shellcheck disable=SC2086
        python src/training/train_tabpfn.py --dataset "$dataset" --external-test $ext
      else
        log "TabPFN: $dataset (IDH)"
        python src/training/train_tabpfn.py --dataset "$dataset"
      fi
    done
  fi
fi

log "All stages finished."
