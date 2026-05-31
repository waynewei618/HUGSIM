#!/usr/bin/env bash

# Project-local download caches. Source this from the repository root.

resource_dir="${HUGSIM_RESOURCE_DIR:-data/resource}"
mkdir -p "${resource_dir}"
resource_dir="$(cd "${resource_dir}" && pwd)"

pixi_cache_dir="${resource_dir}/pixi-cache"

export PIXI_CACHE_DIR="${pixi_cache_dir}"
export PIXI_CACHE_CONDA_PACKAGES_DIR="${pixi_cache_dir}/pkgs"
export PIXI_CACHE_REPODATA_DIR="${pixi_cache_dir}/repodata"
export PIXI_CACHE_PYPI_WHEELS_DIR="${pixi_cache_dir}/uv-cache"
export PIXI_CACHE_PYPI_MAPPING_DIR="${pixi_cache_dir}/http-cache"
export PIP_CACHE_DIR="${resource_dir}/pip-cache"
export UV_CACHE_DIR="${PIXI_CACHE_PYPI_WHEELS_DIR}"
export PIP_FIND_LINKS="${resource_dir}/pip-wheelhouse"
export UV_FIND_LINKS="${resource_dir}/pip-wheelhouse"

mkdir -p \
  "${PIXI_CACHE_DIR}" \
  "${PIXI_CACHE_CONDA_PACKAGES_DIR}" \
  "${PIXI_CACHE_REPODATA_DIR}" \
  "${PIXI_CACHE_PYPI_WHEELS_DIR}" \
  "${PIXI_CACHE_PYPI_MAPPING_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${PIP_FIND_LINKS}"
