# HUGSIM Pixi 环境配置记录

本文记录从进入 Docker 容器 `ubuntu_dev` 开始，按项目 README 配置 HUGSIM `pixi` 环境的过程。

## 前提

- 项目路径在容器内为 `/workspace/HUGSIM`。
- `pixi` 已安装在容器内，版本为 `pixi 0.65.0`。
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

其中 `pytorch3d` 固定到 upstream 的 `stable` tag，`apex` 固定到提交：

```text
ac8214ee6ba77c0037c693828e39d83654d25720
```

必须使用 `--recursive`，否则 `HUGSIM_splat`、`tiny-cuda-nn`、`apex`、`pytorch3d` 等仓库的嵌套依赖可能缺失。例如 `HUGSIM_splat` 缺少 `gsplat/cuda/csrc/third_party/glm` 时，会在编译 `gsplat` 时出现 `glm/glm.hpp` 找不到的问题。

### 3. 安装 pixi 默认环境

`pixi.toml` 已改为从 `external/` 本地路径安装源码依赖，因此初始化 submodule 后直接执行：

```bash
pixi install
```

这一阶段会安装 PyPI/Conda 基础依赖，并从本地 `external/` 编译源码依赖，包括 `gsplat`、`pytorch3d`、`tinycudann`、`simple-knn` 等 CUDA/C++ 扩展。

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
for name in ["gsplat", "pytorch3d", "tinycudann", "apex", "simple_knn"]:
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

## 结果

- HUGSIM 默认 `pixi` 环境已安装完成。
- CUDA 可用。
- 第三方源码库已统一迁移到 `external/` Git submodule。
- `pixi.toml` 已改为通过本地 `external/` path 依赖安装源码包。
- 核心源码扩展包和 `apex` 已成功导入。
- `pixi.lock` 在安装过程中被更新，属于环境解析结果。
