#!/bin/bash
# Bootstrapped Flash mxfp4 launcher with FlyDSL backend.
# Installs flydsl wheel + drops in aiter.ops.flydsl, then starts the server.
set -euo pipefail

WHEEL=/flydsl_setup/flydsl-0.1.3.1+20260418.68f5725-cp310-cp310-manylinux_2_35_x86_64.whl

# Install wheel if not already
if ! python3 -c "import flydsl" 2>/dev/null; then
  echo "Installing FlyDSL wheel..."
  pip install --quiet "$WHEEL" 2>&1 | tail -3
fi

# Drop in aiter.ops.flydsl module
if ! python3 -c "from aiter.ops.flydsl.utils import is_flydsl_available; assert is_flydsl_available()" 2>/dev/null; then
  echo "Dropping in aiter.ops.flydsl module..."
  cp -r /flydsl_src /sgl-workspace/aiter/aiter/ops/flydsl
fi

echo "FlyDSL ready. Booting Flash mxfp4 server..."
exec bash /sgl-pr/launch_dsv4.sh
