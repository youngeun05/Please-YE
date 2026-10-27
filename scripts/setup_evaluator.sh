#!/usr/bin/env bash
# Linux/macOS twin of setup_evaluator.ps1: fetch the pinned MMDetection3D KITTI evaluator.
# Only eval.py from this commit is used (no mmdet3d install, no CUDA build).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/third_party/mmdetection3d"
COMMIT="fe25f7a51d36e3702f961e198894580d83c4387b"
EVAL_FILE="mmdet3d/evaluation/functional/kitti_utils/eval.py"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  # blob-less clone: downloads history metadata only, then the single file we need
  git clone --filter=blob:none --no-checkout https://github.com/open-mmlab/mmdetection3d.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMMIT" -- "$EVAL_FILE"

ACTUAL="$(git -C "$DEST" rev-parse "$COMMIT^{commit}")"
if [ "$ACTUAL" != "$COMMIT" ] || [ ! -f "$DEST/$EVAL_FILE" ]; then
  echo "Evaluator setup failed (commit $ACTUAL, file $DEST/$EVAL_FILE)" >&2
  exit 1
fi
echo "Pinned evaluator ready: $DEST/$EVAL_FILE ($COMMIT)"
