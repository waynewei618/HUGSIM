# HUGSIM 程序运行 Pipeline

## 1. 环境准备

```bash
git submodule update --init --recursive
pixi install
pixi run install-apex
pixi shell
```

## 2. 数据准备

将原始数据集转换为 HUGSIM 训练格式。

```bash
cd data
zsh ./nusc/run.sh      # 或 kitti360 / waymo / pandaset
cd ..
```

## 3. 场景重建

先训练地面 Gaussian，再训练完整场景。

```bash
python train_ground.py \
  --data_cfg ./configs/${dataset_name}.yaml \
  --source_path ${input_path} \
  --model_path ${output_path}

python train.py \
  --data_cfg ./configs/${dataset_name}.yaml \
  --source_path ${input_path} \
  --model_path ${output_path}
```

输出主要包括：

```text
ckpts/ground_chkpnt30000.pth
ckpts/chkpnt30000.pth
ckpts/dynamic_*_chkpnt30000.pth
ckpts/unicycle_*.pth
cfg.yaml
ground_param.pkl
```

## 4. 离线仿真准备

将训练结果整理为仿真可用的场景，并生成可视化 / GUI 配置所需文件。

```bash
python eval_render/export_scene.py \
  --model_path ${recon_scene_path} \
  --output_path ${export_path} \
  --iteration 30000
```

导出后的场景包含：

```text
scene.pth
dynamic_*.pth
unicycle_*.pth
cfg.yaml
ground_param.pkl
meta_data.json
```

然后执行 `convert_scene.py`，作为离线仿真准备的一部分：

```bash
python eval_render/convert_scene.py \
  --model_path ${export_path}
```

该步骤会读取：

```text
${export_path}/cfg.yaml
${export_path}/scene.pth
```

并生成：

```text
${export_path}/vis/semantic.ply
${export_path}/vis/points.ply
${export_path}/vis/scene.splat
```

## 5. 重建效果对比视频

离线仿真准备完成后，可单独生成前视原采集视频和逐帧 3DGS 渲染图像的左右对比，用于观察重建效果。

输入为已导出的场景目录和原始图片所在路径：

```bash
CUDA_VISIBLE_DEVICES=${render_cuda} \
bash render_3dgs/reconstruction_compare/reconstruction_compare.sh \
  ${export_path} \
  ${source_path}
```

其中 `${export_path}` 需要包含：

```text
scene.pth
dynamic_*.pth
cfg.yaml
meta_data.json
```

脚本输出统一写入：

```text
outputs/render_3dgs/<dataset>/<scene>/
```

主要输出包括：

```text
aeb_trajectory.json          # 自车位姿轨迹，含 position、rotation、mileage
aeb_camera.json              # 前视相机内参、外参、分辨率、fps
aeb_front_original.mp4       # 原采集前视视频
aeb_front_compare.mp4        # 左原图、右渲染的对比视频
aeb_real_front_120_rendered.mp4  # 使用 front_120/cam1 实车 AEB 前视内参、near/far、分块渲染和 front.dat LUT 生成的视频
aeb_real_front_120_rendered.timing.csv  # 实车前视每帧渲染耗时，含 LUT 后处理
aeb_trajectory_plots.png     # 里程-高度和水平面轨迹图
```

该步骤只用于重建效果观察，不改变后续 GUI scenario 配置和闭环仿真输入。

## 6. Scenario 配置
车辆模型已完成转换，默认直接使用转换后的车辆模型目录：

```bash
export PATH_3DRealCar=/data/realcar3d
```

已存在的车辆模型输出：

```text
${PATH_3DRealCar}/converted/*.ply
${PATH_3DRealCar}/converted/*.splat
```

不需要重复运行 `eval_render/convert_vehicles.py`。使用 GUI 放置车辆、编辑场景交互配置，并导出 scenario yaml。

```bash
cd gui
python app.py \
  --scene ${export_path} \
  --car_folder ${PATH_3DRealCar}/converted
```

## 7. 闭环仿真

启动 HUGSIM 环境和 AD client，执行闭环仿真。

```bash
CUDA_VISIBLE_DEVICES=${sim_cuda} \
python closed_loop.py \
  --scenario_path ${scenario_yaml} \
  --base_path ./configs/sim/${dataset_name}_base.yaml \
  --camera_path ./configs/sim/${dataset_name}_camera.yaml \
  --kinematic_path ./configs/sim/kinematic.yaml \
  --ad uniad \
  --ad_cuda ${ad_cuda}
```

闭环逻辑：

```text
HUGSimEnv 渲染多相机观测
        ↓
obs_pipe 发送 obs/info 给 AD client
        ↓
AD client 输出 plan_traj
        ↓
plan_pipe 返回规划轨迹
        ↓
traj2control 转换为 acc / steer_rate
        ↓
env.step 推进仿真并检测碰撞、路线完成度
```

## 8. 输出结果

```text
video.mp4      # 六相机拼接视频
data.pkl       # 轨迹、车辆框、规划结果、碰撞信息
infos.pkl      # 每帧环境状态
eval.json      # 仿真评估结果
ground.ply     # 地面点云
scene.ply      # 场景点云
output.txt     # AD client 日志
```

## 简要总结

HUGSIM 先从真实驾驶数据重建 Gaussian 场景，再通过 `export_scene.py` 和 `convert_scene.py` 完成离线仿真准备。之后可单独生成前视原图 / 渲染对比视频观察重建效果，再用 GUI 配置 scenario，最后由 `closed_loop.py` 启动 HUGSimEnv 与 AD client，实现多相机渲染、轨迹规划、车辆控制和闭环评估。
