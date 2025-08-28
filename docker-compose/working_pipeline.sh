#!/usr/bin/env bash
# Start core services from docker-compose-edgex.yml, then launch video-source-x last.

set -euo pipefail

X="${1:-1}"  # choose 1|2|3|4 at run time, default 1

if [[ ! "$X" =~ ^[1-4]$ ]]; then
  echo "Usage: $0 [1|2|3|4]"
  exit 1
fi

FILE="docker-compose-edge${X}.yml"


COMPOSE="docker compose -f ${FILE}"

# Services are assumed to be named exactly like below:
CORE_SERVICES=(redis "object-detector${X}" "object-tracker-stream${X}" "geo-mapper${X}" "feature_extractor${X}")
VIDEO_SERVICE="video-source-stream${X}"

echo "Starting core services for x=${X}: ${CORE_SERVICES[*]}"
$COMPOSE up "${CORE_SERVICES[@]}"

# # Wait for Redis (has a healthcheck in your compose)
# echo -n "Waiting for Redis to be healthy"
# CID="$($COMPOSE ps -q redis || true)"
# while [[ -z "$CID" ]]; do
#   sleep 1
#   CID="$($COMPOSE ps -q redis || true)"
# done
# while [[ "$(
#   docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CID" 2>/dev/null || echo none
# )" != "healthy" ]]; do
#   echo -n "."
#   sleep 1
# done
# echo " ✅"

# # Small buffer for other services to finish init
# sleep 5

# echo "Starting video source service: ${VIDEO_SERVICE}"
# $COMPOSE up "$VIDEO_SERVICE"

# echo "✅ All services up for x=${X}. Video source started last."
