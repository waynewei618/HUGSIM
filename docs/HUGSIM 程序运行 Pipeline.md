# HUGSIM 程序运行 Pipeline

## 1. 环境准备

```bash
git submodule update --init --recursive
pixi install
pixi run install-apex
pixi shell
```

## 2. 数据准备

将原始数据集先转换为统一 loader 输出，再生成训练前产物。下面以 NuScenes 为例，其它数据集使用 `loader/waymo/load.py`、`loader/pandaset/load.py` 或 `loader/me/load.py`。ME 数据集需要先运行 `loader/me/resplit_subscenes.py` 生成 resplit sub-scene，再对单个 sub-scene 执行 loader。

```bash
python loader/nuscenes/load.py \
  --datapath ${raw_dataset_path} \
  --version ${version} \
  --seq ${scene_name} \
  --out ${loader_out}

HUGSIM_DISABLE_XFORMERS=1 python pre_train/run_prepare.py \
  --input ${loader_out} \
  --cuda 0 \
  --total 200000
```

ME 示例：

```bash
python loader/me/resplit_subscenes.py \
  -s 20250317_161633_1 \
  --data-root /mnt/compute-data/e2e/me \
  --output-root outputs/me/resplit \
  --max-frames 150 \
  --max-distance 200 \
  --overlap-frames 0 \
  --min-distance 0.05 \
  --overwrite

python loader/me/load.py \
  --datapath outputs/me/resplit \
  --seq 20250317_161633_1_200m \
  --sub-scene 1
```

## 3. 场景重建

先训练地面 Gaussian，再训练完整场景。

```bash
CUDA_VISIBLE_DEVICES=0 python -u train/run_reconstruction.py \
  --train_cfg ./configs/train.yaml \
  --source_path ${input_path} \
  --model_path ${output_path} \
  --cuda 0
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
python -u train/run_offline_prepare.py \
  --model_path ${output_path} \
  --export_path ${export_path} \
  --iteration 30000
```

如果需要从 loader 输出一键执行 `pre_train`、场景重建和离线仿真准备，可使用根目录脚本：

```bash
bash run_pipeline.sh \
  --input ${loader_out} \
  --output ${output_path} \
  --export ${export_path} \
  --cuda 0 \
  --iteration 30000
```

未显式指定 `--output` 和 `--export` 时，脚本默认在输入目录同级生成 `${输入目录名}_train` 和 `${输入目录名}_export`。

导出后的场景包含：

```text
scene.pth
dynamic_*.pth
unicycle_*.pth
cfg.yaml
ground_param.pkl
meta_data.json
```

`run_offline_prepare.py` 会继续执行 `convert_scene.py`，作为离线仿真准备的一部分：

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

## 5. Scenario 配置
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

## 6. 闭环仿真

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

## 7. 输出结果

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

HUGSIM 先从真实驾驶数据重建 Gaussian 场景，再通过 `export_scene.py` 和 `convert_scene.py` 完成离线仿真准备。之后用 GUI 配置 scenario，最后由 `closed_loop.py` 启动 HUGSimEnv 与 AD client，实现多相机渲染、轨迹规划、车辆控制和闭环评估。
