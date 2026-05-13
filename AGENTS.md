# AGENTS.md instructions for /home/sil/workspace/HUGSIM

## Docker Execution

- 本项目中执行程序（包括 `shell`、`python`、`colmap`、`pixi`、`conda` 等）都应进入 Docker 环境执行。
- 使用的容器名称为 `ubuntu_dev`。
- 在 Docker 中，本项目位于 `/workspace/HUGSIM`。
- 如果 `ubuntu_dev` 容器尚未创建，则先提示用户创建 `ubuntu_dev`，不要直接在宿主机环境执行这些程序。

## GPU Usage

- 本机有两张 GPU 卡；任何需要使用 GPU 的任务，执行前先在 Docker 容器内检查 GPU 占用情况。
- 优先选择空闲或显存占用更低的 GPU，并通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 等方式显式指定使用的 GPU。

## Domestic Download Mirrors

- 下载 GitHub 或其他外网资源时，优先在 Docker 内设置本地代理：
  `export http_proxy="http://127.0.0.1:7890"`
  `export https_proxy="http://127.0.0.1:7890"`
  `export ftp_proxy="http://127.0.0.1:7890"`
- 下载 HuggingFace 权重或模型时，优先设置：
  `export HF_ENDPOINT="https://hf-mirror.com"`
- 使用 `pip` 安装 Python 包时，优先使用清华 PyPI 源：
  `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>`

## Documentation Conventions

- `docs/` 下的文档名即文档主题；内容应围绕该主题展开，不要混入其他主题的通用说明。
- 数据集流程文档（如 Waymo、NuScenes 等）只记录该数据集的输入、运行命令、流程步骤、输出结构和该流程直接读取的路径/变量。
- 环境配置、依赖版本、CUDA/PyTorch/xFormers 适配、pixi 安装、通用模型权重管理等内容应写入 `docs/pixi_environment_setup.md`，不要散落到具体数据集流程文档。
- 规则性约定和长期偏好应写入 `AGENTS.md`，不要写进具体流程文档作为临时 FAQ。
- 编写 Markdown 文档时，环境配置与运行命令默认假设读者已进入 Docker 容器，直接写在容器内执行的命令，不加 `docker exec ubuntu_dev bash -c "..."` 外层包装。
- 文档中路径使用 Docker 容器内路径（如 `/workspace/HUGSIM`、`/workspace/data/...`）。
- 以后凡是修改环境、依赖版本、CUDA/PyTorch/xFormers 适配、模型权重路径、数据路径或运行流程时，都要同步检查 `docs/` 下相关文档是否需要适配更新；如果需要，应在同次变更中更新文档。

## 数据路径约定

- 数据路径不一致时，不要创建软链接作为默认解决方式。
- 优先通过脚本已有参数、环境变量或配置文件覆盖数据路径。
- 如果上游脚本硬编码路径且没有参数入口，优先修改脚本中的路径变量；文档中应记录要修改的变量名和目标路径，而不是要求创建软链接。
