#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEST="$ROOT/deps/mlx-c"
PIN=a26f2d584a14ec619ae8db9a09be03becd47598a
URL=https://github.com/ml-explore/mlx-c.git

if [ -e "$DEST" ]; then
  if [ ! -d "$DEST/.git" ]; then
    echo "refusing non-git existing dependency: $DEST" >&2
    exit 2
  fi
  current=$(git -C "$DEST" rev-parse HEAD)
  if [ "$current" != "$PIN" ]; then
    echo "refusing different mlx-c checkout: $current (expected $PIN)" >&2
    exit 2
  fi
  echo "mlx-c already pinned at $PIN"
  exit 0
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout --detach "$PIN"
echo "mlx-c pinned at $PIN"
