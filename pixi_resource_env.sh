#!/usr/bin/env bash

# Project-local resource paths for pixi-related installs and source builds.
# This script may be sourced manually before `pixi install` / `pixi reinstall`.

_pixi_resource_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${_pixi_resource_repo_root}/data/resource/pip-wheelhouse}"
export UV_FIND_LINKS="${UV_FIND_LINKS:-${_pixi_resource_repo_root}/data/resource/pip-wheelhouse}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${_pixi_resource_repo_root}/.pixi/torch_extensions}"

mkdir -p "${PIP_FIND_LINKS}" "${TORCH_EXTENSIONS_DIR}"
unset _pixi_resource_repo_root
