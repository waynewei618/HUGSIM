# AGENTS.md instructions for /home/sil/workspace/HUGSIM

## 语言与文档

- 默认使用中文回复用户，除非用户明确要求使用英文或其他语言。
- 说明性文档默认使用中文，风格直入主题，先写现象、数据和结论，再写原因分析。
- `AGENTS.md` 只保留项目边界和高频约定；运行流程、参数细节和方案说明优先查看 `docs/` 下的对应文档。

## Docker Execution

- 不要直接在宿主机环境执行项目程序（包括 `shell`、`python`、`pixi`、`conda` 等），默认进入 Docker 容器 `ubuntu_dev`执行。
- 项目容器挂载约定为 `~/workspace:/workspace`、`/data:/data`、`/mnt:/mnt`。
- Git 远程同步操作（`git fetch`、`git pull`、`git push` 等）不进入 Docker，直接在宿主机仓库执行。

## Documentation Conventions

- `docs/` 下的文档名即文档主题；内容应围绕该主题展开，不要混入其他主题的通用说明。
- 说明性文档默认贴近用户当前写作风格：直入主题、少铺垫；公式统一使用 LaTeX 行内公式 `$...$` 或块级公式 `$$...$$`，不要用普通 text 代码块表达数学公式。
- 编写 Markdown 文档时，环境配置与运行命令默认假设读者已进入 Docker 容器，直接写在容器内执行的命令，不加 `docker exec ubuntu_dev bash -c "..."` 外层包装。
- 功能修改后推代码前，要同步检查 `README.md` 和 `docs/` 是否需要更新。