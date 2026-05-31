#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wheelhouse="${repo_root}/data/resource/torch_whl"

mkdir -p "${wheelhouse}"

download_wheel() {
  local filename="$1"
  local url="$2"
  local target="${wheelhouse}/${filename}"
  local partial="${target}.part"

  if [[ -f "${target}" ]]; then
    echo "exists: ${target}"
    return
  fi

  echo "download: ${filename}"
  curl -fL --retry 3 --retry-delay 2 -C - -o "${partial}" "${url}"
  mv "${partial}" "${target}"
}

download_wheel \
  "torch-2.4.1+cu118-cp311-cp311-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu118/torch-2.4.1%2Bcu118-cp311-cp311-linux_x86_64.whl"

download_wheel \
  "torchvision-0.19.1+cu118-cp311-cp311-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu118/torchvision-0.19.1%2Bcu118-cp311-cp311-linux_x86_64.whl"
