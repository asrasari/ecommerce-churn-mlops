#!/usr/bin/env bash
# Build the submission zip (<50 MB), excluding venv, MLflow stores, and lecture notes.
set -euo pipefail

NUMBER="${1:-XXXXXXX}"
OUT="PRJ-goksinbakir-${NUMBER}.zip"
cd "$(dirname "$0")/.."

rm -f "$OUT"
zip -r "$OUT" \
  src airflow/dags airflow/docker-compose.yml airflow/.env \
  tests data requirements.txt README.md conftest.py reports docs scripts \
  -x "*/__pycache__/*" "*.pyc" "*/.pytest_cache/*" \
  >/dev/null

echo "Built $OUT"
du -h "$OUT"
