#!/usr/bin/env bash
# Run this ONCE on Mac after you fork WenjinHou/Uni-OPD into your GitHub account.
# It wires up four submodules: your Uni-OPD fork (mutable), and three pinned upstream repos.

set -euo pipefail

# --- Config -------------------------------------------------------------------
# Override these via environment if your GitHub login isn't `houshihao`.
: "${GH_USER:=houshihao}"
: "${UPSTREAM_OWNER:=WenjinHou}"
: "${UPSTREAM_REPO:=Uni-OPD}"

FORK_URL="https://github.com/${GH_USER}/${UPSTREAM_REPO}.git"
UPSTREAM_URL="https://github.com/${UPSTREAM_OWNER}/${UPSTREAM_REPO}.git"

# Pinned commits — see docs/upstream-cheatsheet.md
SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

cd "$(git rev-parse --show-toplevel)"

# --- 1. Verify the fork exists -----------------------------------------------
echo ">>> Checking that fork ${FORK_URL} exists on GitHub"
if ! gh repo view "${GH_USER}/${UPSTREAM_REPO}" >/dev/null 2>&1; then
  echo "    Fork not found. Creating it now:"
  gh repo fork "${UPSTREAM_OWNER}/${UPSTREAM_REPO}" --clone=false --remote=false
fi

# --- 2. Add the Uni-OPD fork as a submodule ----------------------------------
echo ">>> Adding submodule: third_party/Uni-OPD  ->  ${FORK_URL}"
if [ ! -d third_party/Uni-OPD/.git ] && [ ! -f .gitmodules ] || ! grep -q "third_party/Uni-OPD" .gitmodules 2>/dev/null; then
  git submodule add "${FORK_URL}" third_party/Uni-OPD
fi
# Track upstream as a second remote inside the submodule so you can pull fixes.
cd third_party/Uni-OPD
git remote remove upstream 2>/dev/null || true
git remote add upstream "${UPSTREAM_URL}"
git fetch upstream --quiet
cd ../..

# --- 3. Add Megatron-LM pinned ----------------------------------------------
echo ">>> Adding submodule: third_party/Megatron-LM @ ${MEGATRON_COMMIT}"
if ! grep -q "third_party/Megatron-LM" .gitmodules 2>/dev/null; then
  git submodule add https://github.com/NVIDIA/Megatron-LM.git third_party/Megatron-LM
fi
cd third_party/Megatron-LM
git fetch --tags --quiet
git checkout "${MEGATRON_COMMIT}"
cd ../..

# --- 4. Add sglang pinned ----------------------------------------------------
echo ">>> Adding submodule: third_party/sglang @ ${SGLANG_COMMIT}"
if ! grep -q "third_party/sglang" .gitmodules 2>/dev/null; then
  git submodule add https://github.com/sgl-project/sglang.git third_party/sglang
fi
cd third_party/sglang
git fetch --tags --quiet
git checkout "${SGLANG_COMMIT}"
cd ../..

# --- 5. Add lmms-eval (HEAD) -------------------------------------------------
echo ">>> Adding submodule: third_party/lmms-eval (HEAD)"
if ! grep -q "third_party/lmms-eval" .gitmodules 2>/dev/null; then
  git submodule add https://github.com/EvolvingLMMs-Lab/lmms-eval.git third_party/lmms-eval
fi

# --- 6. Show next steps ------------------------------------------------------
cat <<EOF

>>> Done. Review changes with:

    git status
    git diff --staged .gitmodules

>>> If everything looks right, commit:

    git commit -m "Add upstream submodules (Uni-OPD fork, Megatron-LM, sglang, lmms-eval)"
    git push -u origin main

>>> On the devbox, clone with --recurse-submodules:

    git clone --recurse-submodules <your-mllmopd-repo>

EOF
