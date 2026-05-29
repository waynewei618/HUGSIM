#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."

recreate=0

usage() {
  cat <<'EOF'
Usage:
  scripts/install_pixi_env.sh [--recreate]

Builds the project .pixi environment using project-local download caches.

Options:
  --recreate      Delete .pixi before installing.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recreate)
      recreate=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${recreate}" -eq 1 ]]; then
  rm -rf .pixi
fi

# shellcheck source=scripts/pixi_resource_env.sh
source "${script_dir}/pixi_resource_env.sh"

if [[ ! -f pixi.lock ]]; then
  cat >&2 <<'EOF'
pixi.lock is required for one-pass installs from an empty .pixi environment.

The local path packages under external/ import torch while Pixi resolves PyPI
metadata, so deleting the lock file makes a bare bootstrap unreliable. Restore
pixi.lock or regenerate it from an existing working .pixi environment.
EOF
  exit 2
fi

echo "Using Pixi cache: ${PIXI_CACHE_DIR}"
echo "Using optional PyPI find-links: ${PIP_FIND_LINKS}"
pixi install --locked
