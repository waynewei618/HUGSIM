# HUGSIM Pixi 环境配置记录

本文记录 HUGSIM `pixi` 环境的当前安装规则、本地缓存约定、重建流程和历史排障结果。

## 结论

- 项目命令默认在 Docker 容器 `ubuntu_dev` 的 `/workspace/HUGSIM` 下执行，不在宿主机环境直接跑项目脚本。
- Git 远程同步和 submodule 操作仍在宿主机仓库执行。
- `.pixi/` 是本地运行环境目录，不纳入 Git；环境坏了优先只删 `.pixi/`。
- `pixi.lock` 必须保留并提交。空 `.pixi` 要做到一次 `pixi install`，必须依赖锁文件避开 path 包 metadata 阶段的 PyTorch bootstrap 问题。
- `data/resource/pip-wheelhouse/` 是可选的本地 `find-links` 目录，用来手动放少量已有 wheel，不要求自动填满。
- `data/resource/pixi-cache/` 可作为 Pixi package cache root 使用；是否启用取决于当前 shell 环境变量或容器内 Pixi 全局配置。
- `external/**/build/` 是源码扩展的编译缓存，默认保留；只有怀疑缓存脏了才删。
- PyPI index 使用清华镜像；PyTorch/torchvision 仍使用 `download.pytorch.org` 的 CUDA 11.8 wheel。
- `apex` 不在 `pixi install` 中直接安装，仍通过 `pixi run install-apex` 后置安装。

## 当前前提

- 项目路径在容器内为 `/workspace/HUGSIM`。
- `pixi` 已安装在容器内，当前重建使用版本为 `pixi 0.68.1`。
- 容器内 CUDA 编译工具可用，安装过程中使用的是 CUDA 11.8。
- 所有命令均在 Docker 容器 `ubuntu_dev` 中执行。
- 环境涉及的第三方源码库统一作为 Git submodule 放在 `external/` 目录，并由 `pixi.toml` 通过本地 `path` 依赖引用。

在宿主机中先确认 `ubuntu_dev` 容器存在并正在运行：

```bash
docker ps -a --format '{{.Names}}'
docker inspect -f '{{.State.Running}} {{.Config.WorkingDir}}' ubuntu_dev
```

进入容器：

```bash
docker exec -it ubuntu_dev bash
```

进入项目目录并确认 `pixi` 可用：

```bash
cd /workspace/HUGSIM
pwd
ls -ld /workspace/HUGSIM
command -v pixi
pixi --version
```

## Pixi cache 与本地资源目录

Pixi cache 是下载过的包和解析/构建缓存，默认跨所有 Pixi workspace 和全局工具共享。Pixi 安装环境时不是简单复制 cache，而是优先通过 hard link 或 reflink 从 cache 链接到环境里，从而减少重复占用。

`.pixi/envs/default` 是当前项目的环境目录，不是 package cache。删掉 `.pixi` 只会删当前环境；只要 Pixi cache 和源码 build 缓存还在，`pixi install` 可以快速重建环境。

`external/**/build/` 是本项目 path 源码包的 setuptools/ninja/CUDA 编译缓存，不属于 Pixi cache。`pixi clean cache` 不负责保护或清理这些目录。

`data/resource/pip-wheelhouse/` 是可选的本地 wheel 来源目录，不是自动填充的 cache。只有手动放 wheel，或 `pixi.lock` 明确锁到这些 wheel 时，才会作为离线包来源。

本项目约定的资源目录：

```text
data/resource/pixi-cache/pkgs/       # conda 包缓存，含下载/解包后的 conda packages
data/resource/pixi-cache/repodata/   # conda repodata 缓存
data/resource/pixi-cache/uv-cache/   # PyPI/uv wheel、archive、built-wheel 等缓存
data/resource/pixi-cache/http-cache/ # conda 与 PyPI 名称映射等 HTTP 缓存
data/resource/pip-wheelhouse/        # 手动放置的本地 wheel，可被 find-links 使用
```

Pixi cache root 优先级里，环境变量优先于 config。对本项目最重要的是：

1. `PIXI_CACHE_DIR`
2. `RATTLER_CACHE_DIR`
3. Pixi config 中的 `[cache].root`
4. Pixi 默认 cache 目录

需要让当前 shell 使用项目内 Pixi cache 时，可以显式设置：

```bash
export PIXI_CACHE_DIR=/workspace/HUGSIM/data/resource/pixi-cache
export PIXI_CACHE_CONDA_PACKAGES_DIR=/workspace/HUGSIM/data/resource/pixi-cache/pkgs
export PIXI_CACHE_REPODATA_DIR=/workspace/HUGSIM/data/resource/pixi-cache/repodata
export PIXI_CACHE_PYPI_WHEELS_DIR=/workspace/HUGSIM/data/resource/pixi-cache/uv-cache
export PIXI_CACHE_PYPI_MAPPING_DIR=/workspace/HUGSIM/data/resource/pixi-cache/http-cache
export UV_CACHE_DIR=/workspace/HUGSIM/data/resource/pixi-cache/uv-cache
```

也可以写入当前容器用户的 Pixi 全局配置：

```bash
pixi config set --global cache.root /workspace/HUGSIM/data/resource/pixi-cache
pixi config set --global concurrency.downloads 2
```

全局配置写在当前容器用户的 Pixi config 里，容器重建或清理该 config 后需要重新设置。不要把配置写到 `.pixi/config.toml` 作为唯一来源，因为 `.pixi` 正是允许被删除重建的环境目录。

当前仓库根目录的 `pixi_resource_env.sh` 只设置本地 wheel 查找目录和 PyTorch 扩展编译目录：

```bash
source ./pixi_resource_env.sh
```

脚本导出的变量为：

```bash
PIP_FIND_LINKS=/workspace/HUGSIM/data/resource/pip-wheelhouse
UV_FIND_LINKS=/workspace/HUGSIM/data/resource/pip-wheelhouse
TORCH_EXTENSIONS_DIR=/workspace/HUGSIM/.pixi/torch_extensions
```

这会创建 `data/resource/pip-wheelhouse/` 和 `.pixi/torch_extensions/`，但不会强制修改 Pixi package cache root。检查当前 Pixi cache 配置：

```bash
pixi config list
pixi info | grep 'Cache dir'
```

如果 `pixi info` 显示 `$HOME/.cache/rattler/cache`，说明当前 shell 或 Pixi config 没有把 package cache root 指到 `data/resource/pixi-cache`；这不影响 `pixi.toml` 中 `find-links` 对 `data/resource/pip-wheelhouse` 的使用。

官方参考：

- <https://pixi.prefix.dev/latest/reference/environment_variables/>
- <https://pixi.prefix.dev/latest/reference/pixi_configuration/>
- <https://pixi.prefix.dev/latest/workspace/environment/#caching-packages>
- <https://pixi.prefix.dev/latest/reference/cli/pixi/clean/cache/>

## pixi.toml 结构

当前 `pixi.toml` 的结构要点：

```toml
[dependencies]
python = "==3.11.10"
setuptools = "==78.1.0"
pip = ">=26.1.1,<27"
ninja = ">=1.11.1.4, <2"

[pypi-options]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
find-links = [{ path = "data/resource/pip-wheelhouse" }]
no-build-isolation = ["gsplat", "pytorch3d", "tinycudann", "simple-knn", "xformers"]

[pypi-options.dependency-overrides]
xformers = "*"
```

`ninja` 放在 Conda 依赖中，供 CUDA/C++ 源码扩展构建使用；普通 Python 包保留在 `[pypi-dependencies]`；本地源码包集中放在 `# install from source code` 后面的 path 依赖区。

源码依赖使用本地路径：

```toml
hugsim-env = { path = "./sim", editable = true }
simple-knn = { path = "./external/simple-knn", editable = false }
gsplat = { path = "./external/HUGSIM_splat", editable = false }
flow-vis-torch = { path = "./external/Optical-Flow-Visualization-PyTorch", editable = false }
unidepth = { path = "./external/UniDepth", editable = false }
trajdata = { path = "./external/trajdata", editable = false }
tinycudann = { path = "./external/tiny-cuda-nn/bindings/torch", editable = false }
kitti360Scripts = { path = "./external/kitti360Scripts", editable = false }
simple-waymo-open-dataset-reader = { path = "./external/simple-waymo-open-dataset-reader", editable = false }
pytorch3d = { path = "./external/pytorch3d", editable = false }
nuscenes-devkit = { path = "./external/nuscenes-devkit", editable = false }
xformers = { path = "./external/xformers", editable = false }
```

`external/UniDepth/requirements.txt` 会声明 `xformers>=0.0.26`，但 `xformers` 在本项目中由 `pixi.toml` 显式指定为 `external/xformers` 本地源码路径。`dependency-overrides` 中的 `xformers = "*"` 用于让 Pixi/uv 忽略 UniDepth 传递依赖里的版本范围，直接采用项目显式声明的本地 `external/xformers`，否则 `pixi install --locked` 可能把本地 path 包视为 `0a0.dev0` 并误判锁文件过期。

## 清理边界

只删除当前 workspace 环境：

```bash
rm -rf .pixi
```

保留 `data/resource/pixi-cache` 和 `external/**/build/` 时，下一次 `pixi install` 应复用已下载包和源码编译中间产物。

清理 Pixi 维护过的 cache：

```bash
pixi clean cache
```

如果当前 cache root 指向 `data/resource/pixi-cache`，这会破坏“删除 `.pixi` 后不重新下载”的前提；除非明确要回收磁盘或验证冷启动，不要对本项目执行。

Pixi 支持按类别清理：

```bash
pixi clean cache --pypi       # PyPI/uv 相关缓存
pixi clean cache --conda      # conda package 缓存
pixi clean cache --repodata   # conda repodata 缓存
pixi clean cache --mapping    # conda/PyPI 映射缓存
pixi clean cache --build      # build 相关 Pixi 缓存
pixi clean cache --exec       # pixi exec 环境缓存
```

这些命令只处理 Pixi cache；它们不等价于删除 `external/**/build/`。源码包编译缓存是否清理，仍按 `external/` 源码修改流程单独处理。

## 安装流程

### 1. 初始化 external 子模块

先初始化项目使用的第三方源码库：

```bash
git submodule update --init --recursive
```

这些源码库位于 `external/` 下，包括：

- `external/simple-knn`
- `external/HUGSIM_splat`
- `external/Optical-Flow-Visualization-PyTorch`
- `external/UniDepth`
- `external/trajdata`
- `external/tiny-cuda-nn`
- `external/kitti360Scripts`
- `external/simple-waymo-open-dataset-reader`
- `external/pytorch3d`
- `external/nuscenes-devkit`
- `external/apex`
- `external/ASAP`
- `external/xformers`

其中 `pytorch3d` 固定到 upstream 的 `stable` tag，`apex` 固定到提交：

```text
ac8214ee6ba77c0037c693828e39d83654d25720
```

`xformers` 用于 UniDepth 的 memory efficient attention。当前环境为 `torch 2.4.1+cu118`、CUDA 11.8，需固定到 `v0.0.28.post1` 对应提交：

```text
d3948b5cb9a3711032a0ef0e036e809c7b08c1e0
```

如果 `external/xformers` 是通过 `git submodule update --init --recursive` 初始化的，Git 会自动检出父仓库记录的提交。安装前仍建议显式确认：

```bash
git -C external/xformers describe --tags --always --dirty
git -C external/xformers rev-parse HEAD
```

预期输出包含：

```text
v0.0.28.post1
d3948b5cb9a3711032a0ef0e036e809c7b08c1e0
```

如果是手动 clone 或本地分支状态不对，安装前切到适配 tag：

```bash
git -C external/xformers fetch --tags origin
git -C external/xformers checkout v0.0.28.post1
git -C external/xformers submodule update --init --recursive
```

必须使用 `--recursive`，否则 `HUGSIM_splat`、`tiny-cuda-nn`、`apex`、`pytorch3d` 等仓库的嵌套依赖可能缺失。例如 `HUGSIM_splat` 缺少 `gsplat/cuda/csrc/third_party/glm` 时，会在编译 `gsplat` 时出现 `glm/glm.hpp` 找不到的问题。

### 2. 安装 pixi 默认环境

已有 `pixi.lock` 时，常规安装或 `.pixi` 损坏后的重建直接一次安装：

```bash
cd /workspace/HUGSIM
pixi install
```

需要显式使用本地 wheelhouse 和 PyTorch 扩展编译目录时：

```bash
cd /workspace/HUGSIM
source ./pixi_resource_env.sh
pixi install
```

这一阶段会安装 PyPI/Conda 基础依赖，并从本地 `external/` 编译源码依赖，包括 `gsplat`、`pytorch3d`、`tinycudann`、`simple-knn` 等 CUDA/C++ 扩展。

当前规范不再要求通过注释 `pixi.toml` 后半段源码依赖来分两次安装；常规安装按锁文件一次完成主环境，`apex` 保持后置任务安装。

如果从空 `.pixi` 同时更新锁文件，可能遇到 `ModuleNotFoundError: No module named 'torch'`。这是因为 Pixi 在解析 PyPI path 包 metadata 时还没安装 PyPI torch。本项目不再通过注释 `pixi.toml` 或拆两次 install 处理这个问题；要求是保留 `pixi.lock`，删除 `.pixi` 后直接一次 `pixi install`。

### 3. 安装 apex

按 README 执行：

```bash
pixi run install-apex
```

该任务已改为在 `external/apex` 中执行：

```toml
[tasks]
install-apex = { cmd = "python setup.py install --cuda_ext --cpp_ext", cwd = "external/apex" }
```

安装过程中可能出现 `setup.py install` 弃用警告、`TORCH_CUDA_ARCH_LIST` 未设置警告，以及若干 C++ 编译警告；只要构建结束且后续导入验证通过即可。

## 三种不要从头来的流程

### `.pixi` 损坏

只删除 `.pixi`，不要删除 `data/resource/`，也不要删除 `external/**/build/`：

```bash
rm -rf .pixi
pixi install
pixi run install-apex
```

这样 PyPI/conda 包可复用已配置的 Pixi cache；本地 CUDA/C++ path 包会尽量复用 `external/**/build/` 的中间产物。

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

生产环境不要隐式更新锁文件，使用已有锁文件安装：

```bash
pixi install --locked
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

本次重建中，默认 `pixi install` 曾在 `kornia-rs==0.1.11` 和 `nvidia-cufft-cu11==10.9.0.58` 的下载/解压阶段超时；使用上述参数后安装完成。

## 验证

安装完成后，在容器内执行以下验证：

```bash
pixi run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for name in ["gsplat", "pytorch3d", "tinycudann", "apex", "simple_knn", "xformers"]:
    try:
        mod = __import__(name)
        print(name, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(name, "FAIL", type(e).__name__, e)
PY
```

验证结果：

```text
torch 2.4.1+cu118 cuda 11.8 available True
gsplat OK 1.2.0
pytorch3d OK 0.7.8
tinycudann OK
apex OK
simple_knn OK
xformers OK 0.0.29+d3948b5c.d20260513
```

另外验证 `apex.normalization.FusedLayerNorm` 可导入：

```bash
pixi run python - <<'PY'
import apex, torch
from apex.normalization import FusedLayerNorm
print("apex import OK")
print("torch cuda available", torch.cuda.is_available())
PY
```

输出：

```text
apex import OK
torch cuda available True
```

验证 xFormers 编译产物与当前 CUDA/PyTorch 匹配：

```bash
pixi run python -m xformers.info | egrep "xFormers|build.cuda_version|build.torch_version|memory_efficient_attention.cutlassF|gpu.compute|TORCH_CUDA_ARCH_LIST"
```

预期关键输出：

```text
xFormers 0.0.29+d3948b5c.d20260513
memory_efficient_attention.cutlassF-pt: available
gpu.compute_capability: 8.6
build.cuda_version: 1108
build.torch_version: 2.4.1+cu118
build.env.TORCH_CUDA_ARCH_LIST: 8.6
```

再运行一次最小 CUDA attention 检查：

```bash
pixi run python - <<'PY'
import torch
from xformers.ops import memory_efficient_attention
q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float16)
y = memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
print(tuple(y.shape), y.dtype, y.device)
PY
```

本次也验证过：

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

## PandaSet devkit

PandaSet loader 入口 `loader/pandaset/load.py` 依赖官方 PandaSet devkit 的 `pandaset` Python 包。该包仍按需手动安装，不纳入当前 pixi 锁文件，避免把 devkit 的宽松旧依赖集带入主环境求解。

需要运行 PandaSet loader 时安装：

```bash
pixi run python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'git+https://github.com/scaleapi/pandaset-devkit.git#subdirectory=python'
```

安装后验证：

```bash
pixi run python - <<'PY'
from pandaset import DataSet
print("pandaset import ok")
PY
```

该依赖只用于读取原始 PandaSet 数据并生成 `meta_data.json`、`images/` 等训练前输入。

## xFormers 故障排查

UniDepth 会使用 xFormers 的 memory efficient attention。若深度估计阶段出现 xFormers CUDA 算子不可用，先确认当前安装与 PyTorch/CUDA 匹配：

```bash
pixi run python - <<'PY'
import torch, xformers
from xformers.ops import memory_efficient_attention
print(torch.__version__, torch.version.cuda, xformers.__version__)
q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float16)
y = memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
print(tuple(y.shape), y.dtype, y.device)
PY
```

如果确认为 xFormers CUDA 算子不可用，可以临时设置：

```bash
export HUGSIM_DISABLE_XFORMERS=1
```

这会让 `pre_train/estimate_depth.py` 禁用 UniDepth 的 xFormers 路径，回退到 PyTorch attention。该方式只用于临时绕过环境问题；正常环境应优先修复或重建 `external/xformers`。

## 模型权重

流程中复用的模型权重统一放在项目的 `checkpoints/` 目录下，避免重复下载：

```text
checkpoints/
├── distance_measures_regressor.pth
├── hrnet48_OCR_HMS_IF_checkpoint.pth
├── huggingface/
│   └── hub/
├── torch/
│   └── hub/
│       └── checkpoints/
│           └── alexnet-owt-7be5be79.pth
└── unidepth-v2-vitl14/
    ├── config.json
    └── model.safetensors
```

运行前检查默认权重路径是否存在于 `checkpoints/`。如果要临时使用其他位置，通过对应环境变量覆盖，不建议改回在线下载：

```bash
export INVERSEFORM_MODEL_PATH=/path/to/hrnet48_OCR_HMS_IF_checkpoint.pth
export INVERSEFORM_DISTANCE_MODEL_PATH=/path/to/distance_measures_regressor.pth
export UNIDEPTH_MODEL_PATH=/path/to/unidepth-v2-vitl14
```

训练脚本和 `pre_train/estimate_depth.py` 会分别调用 `train.model_cache.configure_model_cache()`、`pre_train.model_cache.configure_model_cache()`，默认把 `torch.hub`、`HF_HOME`、`HF_HUB_CACHE` 和 `TRANSFORMERS_CACHE` 指向 `/workspace/HUGSIM/checkpoints/` 下的子目录。这样 LPIPS/AlexNet、HuggingFace 模型缓存不会重复下载到 home 目录。

## 结果

- HUGSIM 默认 `pixi` 环境已安装完成。
- CUDA 可用。
- 第三方源码库已统一迁移到 `external/` Git submodule。
- `pixi.toml` 已改为通过本地 `external/` path 依赖安装源码包。
- 包下载和 Pixi/uv 缓存可放到 `data/resource/pixi-cache`；源码扩展编译缓存保留在 `external/**/build/`。
- 已验证删除 `.pixi` 后 `pixi install` 可复用本地缓存完成安装，不重新下载或重新编译。
- 核心源码扩展包、`apex` 和 xFormers CUDA attention 已成功导入或验证。
- `pixi.lock` 在安装过程中被更新，属于环境解析结果。
