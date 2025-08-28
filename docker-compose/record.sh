#!/usr/bin/env bash
# Start video-source-streamX (detached), then run record.py for featureextractor:streamX.

set -euo pipefail

X="${1:-1}"            # 1..4
TIME_LIMIT="${2:-650}" # seconds
FILE="docker-compose-edge${X}.yml"

if [[ ! "$X" =~ ^[1-4]$ ]]; then
  echo "Usage: $0 X [TIME_LIMIT]"
  echo "  X: 1|2|3|4 (stream index)"
  exit 1
fi

echo "Starting video-source-stream${X} from ${FILE}..."
docker compose -f "$FILE" up -d "video-source-stream${X}"

# Optional: brief wait to let the producer connect to Redis
sleep 1

echo "Launching recorder for featureextractor:stream${X} (time limit ${TIME_LIMIT}s)..."
python3 ../tools/sae-introspection/record.py \
  --streams "featureextractor:stream${X}" \
  --time-limit "$TIME_LIMIT"
