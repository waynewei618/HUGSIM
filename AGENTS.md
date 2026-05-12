# AGENTS.md instructions for /home/sil/workspace/HUGSIM

## Docker Execution

- 本项目中执行程序（包括 `shell`、`python`、`colmap`、`pixi`、`conda` 等）都应进入 Docker 环境执行。
- 使用的容器名称为 `ubuntu_dev`。
- 在 Docker 中，本项目位于 `/workspace/HUGSIM`。
- 如果 `ubuntu_dev` 容器尚未创建，则先提示用户创建 `ubuntu_dev`，不要直接在宿主机环境执行这些程序。