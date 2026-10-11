#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:?usage: package_results.sh RESULT_DIR}"
RESULT_DIR="$(readlink -f "$RESULT_DIR")"
PARENT="$(dirname "$RESULT_DIR")"
NAME="$(basename "$RESULT_DIR")"

find "$RESULT_DIR" -type d -name __pycache__ -prune -exec rm -rf -- {} +
find "$RESULT_DIR" -type f -name '*.pyc' -delete

PYTHONNOUSERSITE=1 BENCHMARK_DIR="$RESULT_DIR" python3 "$RESULT_DIR/scripts/create_manifest.py"
PYTHONNOUSERSITE=1 BENCHMARK_DIR="$RESULT_DIR" python3 "$RESULT_DIR/scripts/create_manifest.py" --verify

ARCHIVE="$PARENT/$NAME.publication_ready.tar.gz"
SHA="$ARCHIVE.sha256"
tar -czf "$ARCHIVE" -C "$PARENT" "$NAME"
sha256sum "$ARCHIVE" > "$SHA"
printf '%s\n' "$ARCHIVE" "$SHA"
