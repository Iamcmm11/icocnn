#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[HLS] Build layer0"
cd "$ROOT_DIR/layer0"
make clean
make test_ico_conv

echo "[HLS] Build layer1"
cd "$ROOT_DIR/layer1"
make clean
make test_ico_conv_layer1 test_ico_conv_layer1_debug

echo "[HLS] Build complete"
