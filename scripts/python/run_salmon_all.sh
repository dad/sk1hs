#!/usr/bin/env bash
set -euo pipefail

FASTQ_DIR="fastq"
INDEX="salmon_index"
OUT_DIR="salmon_quant"
THREADS=8

mkdir -p "${OUT_DIR}"

samples=$(
  ls "${FASTQ_DIR}"/*_R1_001.fastq.gz \
  | sed -E 's#.*/##' \
  | sed -E 's/_S[0-9]+_L[0-9]+_R1_001\.fastq\.gz$//' \
  | sort -u
)

for sample in ${samples}; do
  echo "=== Quantifying ${sample} ==="

  r1s=( "${FASTQ_DIR}/${sample}"_S*_L*_R1_001.fastq.gz )
  r2s=( "${FASTQ_DIR}/${sample}"_S*_L*_R2_001.fastq.gz )

  if [[ ! -e "${r1s[0]}" ]]; then
    echo "ERROR: No R1 files found for ${sample}" >&2
    exit 1
  fi
  if [[ ! -e "${r2s[0]}" ]]; then
    echo "ERROR: No R2 files found for ${sample}" >&2
    exit 1
  fi
  if [[ "${#r1s[@]}" -ne "${#r2s[@]}" ]]; then
    echo "ERROR: R1/R2 lane count mismatch for ${sample}: ${#r1s[@]} vs ${#r2s[@]}" >&2
    exit 1
  fi

  if [[ -f "${OUT_DIR}/${sample}/quant.sf" ]]; then
    echo "Skipping ${sample} (already quantified)"
    continue
  fi

  salmon quant \
    -i "${INDEX}" \
    -l A \
    -1 "$(IFS=,; echo "${r1s[*]}")" \
    -2 "$(IFS=,; echo "${r2s[*]}")" \
    --validateMappings \
    --gcBias \
    -p "${THREADS}" \
    -o "${OUT_DIR}/${sample}"
done

echo "All done."
