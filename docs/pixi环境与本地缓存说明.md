# pixi 环境与本地缓存说明

本文档是 HUGSIM 当前 pixi 环境准则。历史重建过程见 `docs/HUGSIM Pixi 环境配置记录.md`；后续新增依赖、重建 `.pixi` 或修改 `external/` 源码时，以本文档为准。

## 基本边界

- 项目命令默认在 Docker 容器 `ubuntu_dev` 的 `/workspace/HUGSIM` 下执行，不在宿主机环境直接跑项目脚本。
- Git 远程同步和 submodule 操作仍在宿主机仓库执行。
- `.pixi/` 是本地运行环境目录，不纳入 Git。
- 包下载和工具缓存统一放在 `data/resource/` 下。当前容器内 Pixi 全局 cache root 已配置为 `/workspace/HUGSIM/data/resource/pixi-cache`，所以直接执行 `pixi install` 也会复用该目录。
- 当前容器内 Pixi 全局下载并发已配置为 `concurrency.downloads = 2`，用于降低大 wheel 下载/解包时的超时和磁盘抖动概率。
- 默认 PyPI index 使用清华镜像；`pixi.lock` 中普通 PyPI wheel URL 已指向 `pypi.tuna.tsinghua.edu.cn/packages/...`。PyTorch/torchvision 仍使用 `download.pytorch.org` 的 CUDA 11.8 wheel。
- `.pixi` 坏了只删 `.pixi`，不要删除 `data/resource/`，也不要删除 `external/**/build/`。
- `data/resource/pip-wheelhouse/` 只是可选的本地 `find-links` 目录，用来手动放少量已有 wheel，不要求自动填满。
- `external/**/build/` 是源码扩展的编译缓存，默认保留；只有怀疑缓存脏了才删。
- `pixi.lock` 必须保留并提交。空 `.pixi` 要做到一次 `pixi install`，必须依赖锁文件避开 path 包 metadata 阶段的 PyTorch bootstrap 问题。

## Pixi cache 是什么

这里按 Pixi 官方文档的定义来区分几个目录：

- Pixi cache 是“下载过的包和解析/构建缓存”，默认跨所有 Pixi workspace 和全局工具共享。官方文档称 Pixi 会缓存此前下载过的 package；安装环境时不是简单复制 cache，而是优先通过 hard link 或 reflink 从 cache 链接到环境里，从而减少重复占用。
- `.pixi/envs/default` 是当前项目的环境目录，不是 package cache。删掉 `.pixi` 只会删当前环境；只要 Pixi cache 和源码 build 缓存还在，`pixi install` 可以快速重建环境。
- `external/**/build/` 是本项目 path 源码包的 setuptools/ninja/CUDA 编译缓存，不属于 Pixi cache。`pixi clean cache` 不负责保护或清理这些目录。
- `data/resource/pip-wheelhouse/` 是可选的本地 wheel 来源目录，不是自动填充的 cache。只有手动放 wheel，或 `pixi.lock` 明确锁到这些 wheel 时，才会作为离线包来源。

当前项目把 Pixi cache root 固定到：

```text
/workspace/HUGSIM/data/resource/pixi-cache
```

对应目录含义按 Pixi 官方 cache kind 可理解为：

```text
data/resource/pixi-cache/pkgs/       # conda 包缓存，含下载/解包后的 conda packages
data/resource/pixi-cache/repodata/   # conda repodata 缓存
data/resource/pixi-cache/uv-cache/   # PyPI/uv wheel、archive、built-wheel 等缓存
data/resource/pixi-cache/http-cache/ # conda 与 PyPI 名称映射等 HTTP 缓存
```

Pixi 官方文档给出的 cache root 优先级里，环境变量优先于 config。对本项目最重要的是：

1. `PIXI_CACHE_DIR`
2. `RATTLER_CACHE_DIR`
3. Pixi config 中的 `[cache].root`
4. Pixi 默认 cache 目录

所以这两种写法都能让后续 `pixi install` 使用 `data/resource/pixi-cache`：

```bash
source scripts/pixi_resource_env.sh
pixi install
```

或使用当前容器中已经设置好的 Pixi 全局配置：

```bash
pixi config set --global cache.root /workspace/HUGSIM/data/resource/pixi-cache
pixi install
```

区别是：`source scripts/pixi_resource_env.sh` 只影响当前 shell，并会强制把本 shell 里的 Pixi cache 指到当前项目的 `data/resource/pixi-cache`；Pixi 全局配置写在当前容器用户的 Pixi config 里，容器重建或清理该 config 后需要重新设置。不要把配置写到 `.pixi/config.toml` 作为唯一来源，因为 `.pixi` 正是允许被删除重建的环境目录。

检查当前 Pixi cache 配置：

```bash
pixi config list
pixi info | grep 'Cache dir'
```

预期包含：

```text
[concurrency]
downloads = 2

/workspace/HUGSIM/data/resource/pixi-cache
```

官方参考：

- <https://pixi.prefix.dev/latest/reference/environment_variables/>
- <https://pixi.prefix.dev/latest/reference/pixi_configuration/>
- <https://pixi.prefix.dev/latest/workspace/environment/#caching-packages>
- <https://pixi.prefix.dev/latest/reference/cli/pixi/clean/cache/>

## 清理边界

Pixi 官方说明里，cache 可以通过 `pixi clean cache` 清理，也可以直接删除 cache 目录；之后 Pixi 会在需要时重新创建 cache。这个行为是“可以清”，不是本项目日常修复环境的推荐操作。

本项目按目标区分：

```bash
rm -rf .pixi
```

只删除当前 workspace 环境。保留 `data/resource/pixi-cache` 和 `external/**/build/` 时，下一次 `pixi install` 应复用已下载包和源码编译中间产物。

```bash
pixi clean cache
```

清理 Pixi 维护过的 cache。当前 cache root 指向 `data/resource/pixi-cache`，所以这会破坏“删除 `.pixi` 后不重新下载”的前提；除非明确要回收磁盘或验证冷启动，不要对本项目执行。

Pixi 支持按类别清理：

```bash
pixi clean cache --pypi       # PyPI/uv 相关缓存
pixi clean cache --conda      # conda package 缓存
pixi clean cache --repodata   # conda repodata 缓存
pixi clean cache --mapping    # conda/PyPI 映射缓存
pixi clean cache --build      # build 相关 Pixi 缓存
pixi clean cache --exec       # pixi exec 环境缓存
```

这些命令只处理 Pixi cache；它们不等价于删除 `external/**/build/`。源码包编译缓存是否清理，仍按本文的 `external/` 源码修改流程单独处理。

## 安装入口

先初始化第三方源码库：

```bash
git submodule update --init --recursive
```

已有 `pixi.lock` 时，常规安装或 `.pixi` 损坏后的重建直接一次安装：

```bash
cd /workspace/HUGSIM
pixi install
pixi run install-apex
```

需要明确删除坏掉的环境目录时：

```bash
rm -rf .pixi
pixi install
pixi run install-apex
```

`scripts/install_pixi_env.sh` 是可选包装，适合想强制按锁文件安装时使用：

```bash
./scripts/install_pixi_env.sh
./scripts/install_pixi_env.sh --recreate
```

它只做四件事：

1. 进入仓库根目录。
2. 加载 `scripts/pixi_resource_env.sh`。
3. 检查 `pixi.lock` 存在。
4. 执行 `pixi install --locked`。

`scripts/pixi_resource_env.sh` 设置：

```bash
PIXI_CACHE_DIR=data/resource/pixi-cache
PIXI_CACHE_CONDA_PACKAGES_DIR=data/resource/pixi-cache/pkgs
PIXI_CACHE_REPODATA_DIR=data/resource/pixi-cache/repodata
PIXI_CACHE_PYPI_WHEELS_DIR=data/resource/pixi-cache/uv-cache
PIXI_CACHE_PYPI_MAPPING_DIR=data/resource/pixi-cache/http-cache
PIP_CACHE_DIR=data/resource/pip-cache
UV_CACHE_DIR=data/resource/pixi-cache/uv-cache
PIP_FIND_LINKS=data/resource/pip-wheelhouse
UV_FIND_LINKS=data/resource/pip-wheelhouse
```

如果 `source` 后执行 `pixi info | grep 'Cache dir'` 仍显示 `$HOME/.cache/rattler/cache`，说明当前 shell 没有加载到本项目脚本，或加载的是旧脚本；此时 `pixi install` 不会使用 `data/resource/pixi-cache`。

## 三种不要从头来的流程

### `.pixi` 损坏

只删除 `.pixi`，不要删除 `data/resource/`，也不要删除 `external/**/build/`：

```bash
rm -rf .pixi
pixi install
pixi run install-apex
```

这样 PyPI/conda 包会复用 `data/resource/pixi-cache`；本地 CUDA/C++ path 包会尽量复用 `external/**/build/` 的中间产物。

### `pixi.toml` 新增或修改依赖

不要删除 `.pixi`，直接增量解析并安装。当前项目的 PyTorch 是 PyPI 包，`gsplat` 等 path 包在 metadata 阶段会 import torch；因此修改依赖时保留已有 `.pixi` 很重要：

```bash
pixi install
```

这会更新 `pixi.lock`，并只安装新增或变化的依赖。之后提交：

```text
pixi.toml
pixi.lock
```

生产环境不要隐式更新锁文件，仍使用：

```bash
./scripts/install_pixi_env.sh
```

如果误删了 `pixi.lock`，先恢复锁文件，不要在空 `.pixi` 状态下重新求解：

```bash
git restore pixi.lock
```

### `external/` 源码修改

`external/` 源码变了但 `pixi.toml` 没变时，不需要更新 `pixi.lock`，也不要跑全量重装。只重装对应 path 包：

```bash
pixi reinstall --locked gsplat
pixi reinstall --locked pytorch3d
pixi reinstall --locked tinycudann
pixi reinstall --locked simple-knn
pixi reinstall --locked xformers
```

不想判断哪个包变了时，重装常见 CUDA 扩展即可：

```bash
pixi reinstall --locked gsplat pytorch3d tinycudann simple-knn xformers
```

只有怀疑 build 缓存脏了，才删对应目录：

```bash
rm -rf external/simple-knn/build \
       external/HUGSIM_splat/build \
       external/tiny-cuda-nn/bindings/torch/build \
       external/pytorch3d/build \
       external/xformers/build

pixi reinstall --locked gsplat pytorch3d tinycudann simple-knn xformers
```

## 一次 pixi install 的原则

常规流程不需要先注释后半段源码包，也不需要手工来回改 `pixi.toml`。

当前 `pixi.toml` 已把本项目源码依赖写成 `external/` 下的 path 包，并通过 `no-build-isolation` 让 CUDA/C++ 扩展使用目标 pixi 环境中的 PyTorch、ninja 和编译工具。已有可用 `pixi.lock` 时，`pixi install --locked` 会按锁文件一次安装主环境和这些 path 包。

如果从空 `.pixi` 同时更新锁文件，可能遇到 `ModuleNotFoundError: No module named 'torch'`。这是因为 Pixi 在解析 PyPI path 包 metadata 时还没安装 PyPI torch。本项目不再通过注释 `pixi.toml` 或拆两次 install 处理这个问题；要求是保留 `pixi.lock`，删除 `.pixi` 后直接一次 `pixi install`。

## 源码依赖注意事项

安装前必须完成递归 submodule 初始化，否则嵌套依赖可能缺失。例如 `HUGSIM_splat` 缺少 `gsplat/cuda/csrc/third_party/glm` 时会在编译 `gsplat` 时找不到 `glm/glm.hpp`。

`xformers` 用于 UniDepth 的 memory efficient attention。当前环境是 `torch 2.4.1+cu118`、CUDA 11.8，`external/xformers` 固定到 `v0.0.28.post1` 对应提交：

```text
d3948b5cb9a3711032a0ef0e036e809c7b08c1e0
```

安装前建议确认：

```bash
git -C external/xformers describe --tags --always --dirty
git -C external/xformers rev-parse HEAD
```

`external/UniDepth/requirements.txt` 声明 `xformers>=0.0.26`，但本项目显式使用 `external/xformers` path 依赖。`dependency-overrides.xformers = "*"` 用于让 pixi/uv 接受项目内 path 版本，避免 `pixi install --locked` 误判锁文件过期。

PandaSet devkit 仍按需手动安装，不纳入当前 pixi 锁文件，避免把 devkit 的宽松旧依赖集带入主环境求解。需要运行 PandaSet loader 时，再按 `docs/HUGSIM Pixi 环境配置记录.md` 中的 PandaSet 小节处理。

`apex` 不在 `pixi install` 中直接安装，仍通过任务安装：

```bash
pixi run install-apex
```

## 网络超时处理

删除 `.pixi` 后重新安装时，如果仍需联网解析或下载，`kornia-rs`、`nvidia-cufft-cu11`、`nvidia-cublas-cu11`、`nvidia-cudnn-cu11`、`open3d` 等大 wheel 可能下载或解压超时。可保留缓存，删除半成品 `.pixi` 后拉长超时：

```bash
rm -rf .pixi
export UV_HTTP_TIMEOUT=1800
export UV_HTTP_RETRIES=10
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.6
pixi install --locked
```

## 验证

本次已验证：

```bash
rm -rf .pixi
time pixi install
```

在 `data/resource/pixi-cache` 和 `external/**/build/` 已存在时，裸 `pixi install` 直接完成，没有重新下载或重新编译。当前缓存规模约：

```text
data/resource/                                   11G
.pixi/                                          101M（仅 pixi install；install-apex 后约 168M）
external/HUGSIM_splat/build/                    40M
external/pytorch3d/build/                       49M
external/xformers/build/                        64M
external/tiny-cuda-nn/bindings/torch/build/     31M
external/simple-knn/build/                      5.3M
external/apex/build/                            32M
```

基础验证：

```bash
pixi run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for name in ["gsplat", "pytorch3d", "tinycudann", "simple_knn", "xformers"]:
    mod = __import__(name)
    print(name, "OK", getattr(mod, "__version__", ""))
PY
```

验证 `apex`：

```bash
pixi run python - <<'PY'
from apex.normalization import FusedLayerNorm
print("apex OK")
PY
```

验证 xFormers CUDA attention：

```bash
pixi run python -m xformers.info | egrep "xFormers|build.cuda_version|build.torch_version|memory_efficient_attention.cutlassF|gpu.compute|TORCH_CUDA_ARCH_LIST"
```

如果深度估计阶段确认为 xFormers CUDA 算子不可用，可以临时设置：

```bash
export HUGSIM_DISABLE_XFORMERS=1
```

这只用于临时绕过环境问题；正常环境应优先修复或重建 `external/xformers`。
