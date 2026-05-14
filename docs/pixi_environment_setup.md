# HUGSIM Pixi 环境配置记录

本文记录从进入 Docker 容器 `ubuntu_dev` 开始，按项目 README 配置 HUGSIM `pixi` 环境的过程。

## 前提

- 项目路径在容器内为 `/workspace/HUGSIM`。
- `pixi` 已安装在容器内，当前重建使用版本为 `pixi 0.68.1`。
- 容器内 CUDA 编译工具可用，安装过程中使用的是 CUDA 11.8。
- 所有命令均在 Docker 容器 `ubuntu_dev` 中执行。
- 环境涉及的第三方源码库统一作为 Git submodule 放在 `external/` 目录，并由 `pixi.toml` 通过本地 `path` 依赖引用。

## 配置步骤

### 1. 进入 Docker 容器

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

### 2. 初始化 external 子模块

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

### 3. 安装 pixi 默认环境

`pixi.toml` 已改为从 `external/` 本地路径安装源码依赖，因此初始化 submodule 后直接执行：

```bash
pixi install
```

这一阶段会安装 PyPI/Conda 基础依赖，并从本地 `external/` 编译源码依赖，包括 `gsplat`、`pytorch3d`、`tinycudann`、`simple-knn` 等 CUDA/C++ 扩展。

如果删除 `.pixi` 后重新安装时，遇到 `kornia-rs`、`nvidia-cufft-cu11`、`nvidia-cublas-cu11`、`nvidia-cudnn-cu11`、`open3d` 等大 wheel 下载或解压超时，可以保留已下载的 pixi/uv 缓存，重新删除半成品 `.pixi` 后降低并发并拉长超时时间：

```bash
rm -rf .pixi
export UV_HTTP_TIMEOUT=1800
export UV_HTTP_RETRIES=10
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.6
pixi install --frozen --concurrent-downloads 2
```

本次重建中，默认 `pixi install` 曾在 `kornia-rs==0.1.11` 和 `nvidia-cufft-cu11==10.9.0.58` 的下载/解压阶段超时；使用上述参数后安装完成。

对应的 `pixi.toml` 源码依赖使用本地路径：

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

这会让 `data/utils/estimate_depth.py` 禁用 UniDepth 的 xFormers 路径，回退到 PyTorch attention。该方式只用于临时绕过环境问题；正常环境应优先修复或重建 `external/xformers`。

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

训练脚本和 `data/utils/estimate_depth.py` 会调用 `utils.model_cache.configure_model_cache()`，默认把 `torch.hub`、`HF_HOME`、`HF_HUB_CACHE` 和 `TRANSFORMERS_CACHE` 指向 `/workspace/HUGSIM/checkpoints/` 下的子目录。这样 LPIPS/AlexNet、HuggingFace 模型缓存不会重复下载到 home 目录。

## 结果

- HUGSIM 默认 `pixi` 环境已安装完成。
- CUDA 可用。
- 第三方源码库已统一迁移到 `external/` Git submodule。
- `pixi.toml` 已改为通过本地 `external/` path 依赖安装源码包。
- 核心源码扩展包、`apex` 和 xFormers CUDA attention 已成功导入或验证。
- `pixi.lock` 在安装过程中被更新，属于环境解析结果。
