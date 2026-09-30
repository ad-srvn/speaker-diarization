#!/bin/sh
set -eu

cd "$(dirname "$0")"

PYTHON_BIN=""
for candidate in python3.14 python3.13 python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(not ((3, 12) <= sys.version_info[:2] <= (3, 14)))' 2>/dev/null; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.12, 3.13, or 3.14 is required. On macOS, install it with: brew install python"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. On macOS, install it with: brew install ffmpeg"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating the application environment..."
  "$PYTHON_BIN" -m venv .venv
fi

if [ ! -f .venv/.requirements-installed ] || [ requirements.txt -nt .venv/.requirements-installed ]; then
  echo "Installing application dependencies (first launch can take several minutes)..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.requirements-installed
fi

exec .venv/bin/python app.py "$@"
