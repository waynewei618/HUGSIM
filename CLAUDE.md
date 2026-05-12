# Docker Execution

- 本项目中执行程序（包括 `shell`、`python`、`colmap`、`pixi`、`conda` 等）都应进入 Docker 环境执行。
- 使用的容器名称为 `ubuntu_dev`。
- 在 Docker 中，本项目位于 `/workspace/HUGSIM`。
- 如果 `ubuntu_dev` 容器尚未创建，则先提示用户创建 `ubuntu_dev`，不要直接在宿主机环境执行这些程序。


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

- 编写 Markdown 文档时，环境配置与运行命令默认假设读者已进入 Docker 容器，直接写在容器内执行的命令，不加 `docker exec ubuntu_dev bash -c "..."` 外层包装。
- 文档中路径使用 Docker 容器内路径（如 `/workspace/HUGSIM`、`/workspace/data/...`）。
- 以后凡是修改环境、依赖版本、CUDA/PyTorch/xFormers 适配、模型权重路径、数据路径或运行流程时，都要同步检查 `docs/` 下相关文档是否需要适配更新；如果需要，应在同次变更中更新文档。
