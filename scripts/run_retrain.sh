#!/usr/bin/env bash
set -euo pipefail
python scripts/export_training_data.py
cd recsys && python -m recsys.pipelines.train --config config/default.yaml
curl -X POST http://localhost:8000/api/v1/admin/reload-model
