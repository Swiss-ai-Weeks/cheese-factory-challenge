#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$SCRIPT_DIR/docker-compose.yml"
DURATION=${1:-75}
OUTPUT=${2:-docs/evidence/p23-live-showcase.mp4}

if [[ ! "$DURATION" =~ ^[0-9]+$ ]] || ((DURATION < 15 || DURATION > 180)); then
  echo "duration must be an integer from 15 through 180 seconds" >&2
  exit 2
fi
if [[ "$OUTPUT" = /* || "/$OUTPUT/" == *"/../"* || "$OUTPUT" != *.mp4 ]]; then
  echo "output must be a repository-relative .mp4 path without '..'" >&2
  exit 2
fi

"$SCRIPT_DIR/factory-demo.sh" replay
container=$(docker compose -p isim -f "$COMPOSE" ps -q remote-desktop)
[[ -n "$container" ]] || { echo "remote desktop container is unavailable" >&2; exit 1; }

container_output=/tmp/cheese-factory-showcase.mp4
docker exec "$container" ffmpeg -y -loglevel warning \
  -f x11grab -draw_mouse 0 -framerate 20 -video_size 1920x1080 -i :99 \
  -t "$DURATION" -vf scale=1280:720 -c:v libx264 -preset veryfast -crf 28 \
  -pix_fmt yuv420p -movflags +faststart "$container_output"

install -d "$(dirname "$PROJECT_ROOT/$OUTPUT")"
temporary="$PROJECT_ROOT/$OUTPUT.tmp"
docker cp "$container:$container_output" "$temporary"
mv -f "$temporary" "$PROJECT_ROOT/$OUTPUT"
docker exec "$container" rm -f "$container_output"

printf 'DEMO_CAPTURE=%s\n' "$OUTPUT"
sha256sum "$PROJECT_ROOT/$OUTPUT"
