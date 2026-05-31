# HUGSIM Pixi 环境配置记录

本文记录 HUGSIM `pixi` 环境的当前安装规则、重建流程和历史排障结果。

## 结论

- 项目命令默认在 Docker 容器 `ubuntu_dev` 的 `/workspace/HUGSIM` 下执行，不在宿主机环境直接跑项目脚本。
- Git 远程同步和 submodule 操作仍在宿主机仓库执行。
- `.pixi/` 是本地运行环境目录，不纳入 Git；环境坏了优先只删 `.pixi/`。
- `pixi.lock` 必须保留并提交。空 `.pixi` 要做到一次 `pixi install`，必须依赖锁文件避开 path 包 metadata 阶段的 PyTorch bootstrap 问题。
- `torch`、`torchvision` 通过本地 wheel 安装；wheel 文件放在 `data/resource/torch_whl/`，由 `ensure_torch_wheels.sh` 提前下载。
- PyPI index 使用清华镜像；`torch`、`torchvision` 在 `pixi.toml` 中固定为 `data/resource/torch_whl/` 下的 CUDA 11.8 本地 wheel。
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

## 本地 Torch Wheel

先下载 `torch`、`torchvision` 两个 CUDA 11.8 wheel：

```bash
cd /workspace/HUGSIM
bash ./ensure_torch_wheels.sh
```

脚本会把文件放到：

```text
data/resource/torch_whl/
├── torch-2.4.1+cu118-cp311-cp311-linux_x86_64.whl
└── torchvision-0.19.1+cu118-cp311-cp311-linux_x86_64.whl
```

脚本不做 sha256 校验；如果已有 wheel 文件损坏，手动删除对应 `.whl` 后重新运行脚本。

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
no-build-isolation = ["gsplat", "pytorch3d", "tinycudann", "simple-knn", "xformers"]

[pypi-options.dependency-overrides]
xformers = { path = "./external/xformers" }

```

`ninja` 放在 Conda 依赖中，供 CUDA/C++ 源码扩展构建使用；普通 Python 包保留在 `[pypi-dependencies]`；本地源码包集中放在 `# install from source code` 后面的 path 依赖区。

`torch`、`torchvision` 固定使用本地 wheel 路径：

```toml
torch = { path = "data/resource/torch_whl/torch-2.4.1+cu118-cp311-cp311-linux_x86_64.whl" }
torchvision = { path = "data/resource/torch_whl/torchvision-0.19.1+cu118-cp311-cp311-linux_x86_64.whl" }
```

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

`external/UniDepth/requirements.txt` 会声明 `xformers>=0.0.26`，但 `xformers` 在本项目中由 `pixi.toml` 显式指定为 `external/xformers` 本地源码路径。`dependency-overrides` 也必须指向同一个本地路径，否则 Pixi 可能改选 PyPI 上的预编译 xFormers wheel，导致 CUDA/PyTorch 版本与本项目的本地 torch wheel 不匹配。

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

### 2. 下载本地 torch wheel

```bash
cd /workspace/HUGSIM
bash ./ensure_torch_wheels.sh
```

### 3. 安装 pixi 默认环境

已有 `pixi.lock` 时，常规安装或 `.pixi` 损坏后的重建直接一次安装：

```bash
cd /workspace/HUGSIM
pixi install
```

这一阶段会安装 PyPI/Conda 基础依赖，并从本地 `external/` 编译源码依赖，包括 `gsplat`、`pytorch3d`、`tinycudann`、`simple-knn` 等 CUDA/C++ 扩展。

当前规范不再要求通过注释 `pixi.toml` 后半段源码依赖来分两次安装；常规安装按锁文件一次完成主环境，`apex` 保持后置任务安装。

如果从空 `.pixi` 同时更新锁文件，可能遇到 `ModuleNotFoundError: No module named 'torch'`。这是因为 Pixi 在解析 PyPI path 包 metadata 时还没安装本地 torch wheel。已有 `pixi.lock` 的常规重建不再通过注释 `pixi.toml` 或拆两次 install 处理这个问题；要求是保留 `pixi.lock`，删除 `.pixi` 后直接一次 `pixi install`。

#### 没有 pixi.lock 时的兜底流程

确实没有可用 `pixi.lock`，必须从 `pixi.toml` 重新生成锁文件时，使用两段式安装。原因是本项目多个本地 `path` 源码包会在 metadata 或 build 阶段 import `torch`，而空 `.pixi` 首次求解时本地 torch wheel 尚未安装进环境，容易提前失败。

第一步，临时注释掉 `pixi.toml` 中 `# install from source code` 下面的本地源码包：

```toml
# install from source code
# hugsim-env = { path = "./sim", editable = true }
# simple-knn = { path = "./external/simple-knn", editable = false }
# gsplat = { path = "./external/HUGSIM_splat", editable = false }
# flow-vis-torch = { path = "./external/Optical-Flow-Visualization-PyTorch", editable = false }
# unidepth = { path = "./external/UniDepth", editable = false }
# trajdata = { path = "./external/trajdata", editable = false }
# tinycudann = { path = "./external/tiny-cuda-nn/bindings/torch", editable = false }
# kitti360Scripts = { path = "./external/kitti360Scripts", editable = false }
# simple-waymo-open-dataset-reader = { path = "./external/simple-waymo-open-dataset-reader", editable = false }
# pytorch3d = { path = "./external/pytorch3d", editable = false }
# nuscenes-devkit = { path = "./external/nuscenes-devkit", editable = false }
# xformers = { path = "./external/xformers", editable = false }
```

然后安装基础环境并生成基础锁文件：

```bash
cd /workspace/HUGSIM
bash ./ensure_torch_wheels.sh
pixi install
```

第二步，恢复上面这些源码包，再运行：

```bash
pixi install
```

这样第二次安装源码包时，`torch`、CUDA 运行库、`ninja` 等构建依赖已经在 pixi 环境中。生成成功后提交新的锁文件：

```text
pixi.toml
pixi.lock
```

完成主环境后，仍按后续步骤安装 `apex`：

```bash
pixi run install-apex
```

### 4. 安装 apex

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

只删除 `.pixi`，不要删除 `data/resource/torch_whl/`：

```bash
rm -rf .pixi
bash ./ensure_torch_wheels.sh
pixi install
pixi run install-apex
```

### `pixi.toml` 新增或修改依赖

不要删除 `.pixi`，直接增量解析并安装。当前项目的 PyTorch 来自本地 wheel，`gsplat` 等 path 包在 metadata 阶段会 import torch；因此修改依赖时保留已有 `.pixi` 很重要：

```bash
bash ./ensure_torch_wheels.sh
pixi install
```

这会更新 `pixi.lock`，并只安装新增或变化的依赖。之后提交：

```text
pixi.toml
pixi.lock
```

生产环境不要隐式更新锁文件，使用已有锁文件安装：

```bash
bash ./ensure_torch_wheels.sh
pixi install --locked
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

## 网络超时处理

删除 `.pixi` 后重新安装时，如果仍需联网解析或下载，`kornia-rs`、`nvidia-cufft-cu11`、`nvidia-cublas-cu11`、`nvidia-cudnn-cu11`、`open3d` 等大 wheel 可能下载或解压超时。可拉长超时：

```bash
rm -rf .pixi
export UV_HTTP_TIMEOUT=1800
export UV_HTTP_RETRIES=10
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.6
bash ./ensure_torch_wheels.sh
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

## 结果

- HUGSIM 默认 `pixi` 环境已安装完成。
- CUDA 可用。
- 第三方源码库已统一迁移到 `external/` Git submodule。
- `pixi.toml` 已改为通过本地 `external/` path 依赖安装源码包。
- `torch`、`torchvision` 已改为通过 `data/resource/torch_whl/` 下的本地 wheel 安装。
- 核心源码扩展包、`apex` 和 xFormers CUDA attention 已成功导入或验证。
- `pixi.lock` 在安装过程中被更新，属于环境解析结果。
