#!/usr/bin/env bash
# Pull latest from origin and refresh submodules on the devbox.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch --all --prune
git pull --ff-only --recurse-submodules
git submodule update --init --recursive
git submodule foreach 'echo "  $name  @  $(git rev-parse --short HEAD)"'
