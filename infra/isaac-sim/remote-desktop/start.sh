#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:99

Xvfb :99 -screen 0 1920x1080x24 -ac -nolisten tcp &
openbox &

chrome_path="$(find /ms-playwright -type f -path '*/chrome-linux*/chrome' | head -n 1)"
if [[ -z "$chrome_path" ]]; then
    echo "Chromium executable not found" >&2
    exit 1
fi

"$chrome_path" \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-features=Translate,PasswordManagerOnboarding \
    --no-first-run \
    --start-maximized \
    --app="http://127.0.0.1:${WEB_VIEWER_PORT}" \
    --user-data-dir=/tmp/isaac-browser-profile &

x11vnc \
    -display :99 \
    -localhost \
    -forever \
    -shared \
    -nopw \
    -rfbport 5900 &

exec websockify --web=/usr/share/novnc 0.0.0.0:6080 127.0.0.1:5900
