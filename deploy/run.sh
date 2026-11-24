#!/usr/bin/env bash
# Run a push and keep the console output in logs/push_<date>.log
#   ./deploy/run.sh --silver s3://…/UPC=N_02001_19004_DL            (dry run)
#   ./deploy/run.sh --silver s3://…/UPC=N_02001_19004_DL --apply
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"
mkdir -p logs
"$APP_DIR/venv/bin/python" push.py "$@" 2>&1 | tee -a "logs/push_$(date +%F).log"
exit "${PIPESTATUS[0]}"
