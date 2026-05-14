# AGENTS.md instructions for /home/sil/workspace/HUGSIM

## Docker Execution

- 本项目中执行程序（包括 `shell`、`python`、`colmap`、`pixi`、`conda` 等）都应进入 Docker 环境执行。
- 使用的容器名称为 `ubuntu_dev`。
- 在算力机和控制机上，本项目目录统一为：宿主机 `~/workspace/HUGSIM`，Docker 容器内 `/workspace/HUGSIM`。
- 如果 `ubuntu_dev` 容器尚未创建，则先提示用户创建 `ubuntu_dev`，不要直接在宿主机环境执行这些程序。
- 代码同步是例外：`git fetch`、`git pull`、`git push` 等 Git 远程同步操作不进入 Docker，直接在宿主机仓库执行。
- 数据同步是例外：控制机与算力机之间使用 `ssh`、`rsync` 同步数据时，不需要进入 Docker 容器，应直接从控制机宿主机发起。
- 如果控制机当前工作区或 Docker 路径下缺少数据，优先尝试从算力机同步过来；运行程序和文档命令仍默认使用 Docker 内路径，例如 `/workspace/data/...`、`/workspace/HUGSIM/...`。

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

## 阶段目标：AEB HIL/VIL 前视渲染

- 当前阶段目标是基于离线仿真准备阶段输出的结果，提供自车世界位姿、实际相机内参、实际相机相对于自车的安装外参和分辨率，渲染得到前视视频，用于 AEB 测试的 HIL/VIL 实际项目。
- 该阶段相关代码集中放在 `aeb_hil_vil_render/`，不要继续散落到通用训练、数据预处理或评测脚本目录中。
- 实际项目开发的程序应功能明确单一，命令输入保持简单；主渲染程序 `render_front_view.py` 只接受 `scene_path`、`trajectory.json`、`camera.json` 三个输入。
- `trajectory.json` 只放自车位姿轨迹的最少信息，核心字段为每帧 `ego_position`、自车旋转和从 0 开始累计的 `mileage`；自车旋转可用 `ego_quaternion_wxyz`、`ego_quaternion_xyzw` 或 `ego_rpy` 表达，实际项目不能只给 3D 位置。`camera.json` 放实际相机标定信息，核心字段为 `intrinsics`、`camera_to_ego`、`width`、`height` 和必要的 `fps`。
- 训练场景示例信息提取放在 `extract_scene_inputs.py`，渲染后与原采集视频合成对比放在 `compose_compare_video.py`，不要混入主渲染入口。
- 相机内外参必须来自实际相机标定，不要在实际项目代码中用默认相机参数、虚构外参或仅 3D 位置替代完整相机外参；从训练场景提取的示例 `camera_to_ego` 只能用于复现示例，实际项目必须替换为真实安装外参。
- 重建效果观察生成的 `aeb_*.json`、`aeb_*.mp4` 和轨迹观察图片不要写回训练场景目录，默认集中放到 `outputs/aeb_hil_vil_render/<dataset>/<scene>/`。轨迹观察图片固定为 `aeb_trajectory_plots.png`，左侧里程-高度，右侧水平面 x-y；当前 HUGSIM 场景坐标中高度使用 scene `y`，水平面 x-y 使用 `(scene z, -scene x)`。
