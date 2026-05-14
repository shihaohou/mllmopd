#!/usr/bin/env bash
# Switch the Uni-OPD submodule from upstream (WenjinHou) to your own fork.
# Run this ONCE on Mac when you start needing to commit changes to OPD reward /
# loss / data — i.e., when read-only access to upstream is no longer enough.
#
# Workflow:
#   1. Fork WenjinHou/Uni-OPD on github.com (UI: click Fork). Forks to your_login/Uni-OPD.
#   2. GH_USER=your_login bash scripts/init/switch_to_fork.sh
#   3. git commit + push
#   4. On the devbox: git pull --recurse-submodules && git submodule sync && \
#        git submodule update --init --recursive

set -euo pipefail

: "${GH_USER:?set GH_USER to your GitHub login, e.g. GH_USER=shihaohou bash $0}"
FORK_URL="https://github.com/${GH_USER}/Uni-OPD.git"
UPSTREAM_URL="https://github.com/WenjinHou/Uni-OPD.git"

cd "$(git rev-parse --show-toplevel)"

# Optional: verify the fork actually exists
if command -v curl >/dev/null; then
  if ! curl -sf -o /dev/null "https://github.com/${GH_USER}/Uni-OPD"; then
    echo "ERROR: ${FORK_URL} is not reachable. Fork it on github.com first."
    exit 1
  fi
fi

# 1. Point .gitmodules at the fork
git config -f .gitmodules submodule.third_party/Uni-OPD.url "${FORK_URL}"
git submodule sync third_party/Uni-OPD

# 2. Inside the submodule: origin -> fork, upstream -> WenjinHou
cd third_party/Uni-OPD
git remote set-url origin "${FORK_URL}"
git remote remove upstream 2>/dev/null || true
git remote add upstream "${UPSTREAM_URL}"
git fetch upstream --quiet
cd ../..

echo ">>> Uni-OPD submodule now points at ${FORK_URL}"
echo ">>> Inside third_party/Uni-OPD: origin=fork, upstream=${UPSTREAM_URL}"
echo
echo ">>> Commit the change:"
echo "    git add .gitmodules third_party/Uni-OPD"
echo "    git commit -m 'Switch Uni-OPD submodule to ${GH_USER} fork'"
echo "    git push"
