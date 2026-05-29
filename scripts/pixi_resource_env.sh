#!/usr/bin/env bash

# Project-local download caches. Source this from the repository root.

resource_dir="${HUGSIM_RESOURCE_DIR:-data/resource}"
mkdir -p "${resource_dir}"
resource_dir="$(cd "${resource_dir}" && pwd)"

export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${resource_dir}/pixi-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${resource_dir}/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${resource_dir}/uv-cache}"
export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${resource_dir}/pip-wheelhouse}"
export UV_FIND_LINKS="${UV_FIND_LINKS:-${resource_dir}/pip-wheelhouse}"

mkdir -p "${PIXI_CACHE_DIR}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${PIP_FIND_LINKS}"
