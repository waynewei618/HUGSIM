#!/bin/bash
set -e

# HUGSIM 一键式运行预处理 (pre_train)、场景重建 (train) 与离线仿真准备脚本

# 默认参数
CUDA_DEV="0"
PYTHON_BIN=".pixi/envs/default/bin/python"
OFFLINE_ITERATION="30000"

usage() {
    local exit_code="${1:-1}"
    echo "使用说明:"
    echo "  $0 --input <loader输出目录> [选项]"
    echo ""
    echo "选项:"
    echo "  --input <dir>       (必填) loader的输出文件夹路径"
    echo "  --output <dir>      (选填) 重建训练结果输出路径，默认与输入目录同级且以 _train 结尾"
    echo "  --export <dir>      (选填) 离线仿真导出路径，默认与输入目录同级且以 _export 结尾"
    echo "  --cuda <device_id>  (选填) CUDA GPU设备ID，默认: 0"
    echo "  --iteration <n>     (选填) 导出使用的训练迭代数，默认: 30000"
    echo "  -h, --help          显示帮助信息"
    exit "$exit_code"
}

# 解析参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_DIR="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        --export) EXPORT_DIR="$2"; shift ;;
        --cuda) CUDA_DEV="$2"; shift ;;
        --iteration) OFFLINE_ITERATION="$2"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "未知参数: $1"; usage ;;
    esac
    shift
done

# 校验输入
if [ -z "$INPUT_DIR" ]; then
    echo "错误: 必须使用 --input 指定loader的输出文件夹。"
    usage
fi

# 转化为绝对路径
INPUT_DIR=$(readlink -f "$INPUT_DIR" 2>/dev/null || realpath "$INPUT_DIR")

if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入文件夹不存在: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$INPUT_DIR/meta_data.json" ]; then
    echo "错误: 输入目录中未找到 meta_data.json，请确认其是否为正确的loader输出目录。"
    exit 1
fi

# 获取父目录和基本名，用以派生默认输出路径
PARENT_DIR=$(dirname "$INPUT_DIR")
DIR_NAME=$(basename "$INPUT_DIR")

if [ -z "$OUTPUT_DIR" ]; then
    PRE_TRAIN_DIR="$PARENT_DIR/${DIR_NAME}_pre_train"
    TRAIN_DIR="$PARENT_DIR/${DIR_NAME}_train"
    DEFAULT_EXPORT_DIR="$PARENT_DIR/${DIR_NAME}_export"
else
    # 如果用户显式指定了 --output，则将训练路径设为用户指定的值，预处理路径设为其同级的 _pre_train 后缀目录
    OUTPUT_DIR_ABS=$(readlink -f "$OUTPUT_DIR" 2>/dev/null || realpath "$OUTPUT_DIR")
    TRAIN_DIR="$OUTPUT_DIR_ABS"
    PRE_TRAIN_DIR="${OUTPUT_DIR_ABS}_pre_train"
    DEFAULT_EXPORT_DIR="${OUTPUT_DIR_ABS}_export"
fi

if [ -z "$EXPORT_DIR" ]; then
    EXPORT_DIR="$DEFAULT_EXPORT_DIR"
else
    EXPORT_DIR=$(readlink -f "$EXPORT_DIR" 2>/dev/null || realpath "$EXPORT_DIR")
fi

echo "========================================================="
echo "   HUGSIM Pipeline 一键启动"
echo "========================================================="
echo " 输入路径 (Loader Out) : $INPUT_DIR"
echo " 预处理路径 (Pre-train): $PRE_TRAIN_DIR"
echo " 训练输出路径 (Train)  : $TRAIN_DIR"
echo " 离线导出路径 (Export) : $EXPORT_DIR"
echo " 使用 GPU 设备         : $CUDA_DEV"
echo " 导出迭代数            : $OFFLINE_ITERATION"
echo " Python 解释器        : $PYTHON_BIN"
echo "========================================================="

# 准备预处理专用的工作目录（由于 pre_train 必须在输入目录下原地操作，我们先复制一份，保证原始输入干净）
if [ ! -d "$PRE_TRAIN_DIR" ]; then
    echo "正在创建预处理工作目录，并将原始输入复制到: $PRE_TRAIN_DIR ..."
    mkdir -p "$PRE_TRAIN_DIR"
    cp -r "$INPUT_DIR"/. "$PRE_TRAIN_DIR"/
else
    echo "预处理工作目录已存在: $PRE_TRAIN_DIR (如有必要，请手动清理)"
fi

# 确保训练输出目录存在
mkdir -p "$TRAIN_DIR"

# 1. 运行 pre_train 预处理阶段
echo "Step 1/2: 正在运行 pre_train 预处理阶段..."
HUGSIM_DISABLE_XFORMERS=1 CUDA_VISIBLE_DEVICES=$CUDA_DEV "$PYTHON_BIN" pre_train/run_prepare.py \
    --input "$PRE_TRAIN_DIR" \
    --cuda "$CUDA_DEV" \
    --total 200000

echo "pre_train 阶段运行完成！"
echo "---------------------------------------------------------"

# 2. 运行 train 重建阶段与离线仿真准备阶段
echo "Step 2/2: 正在运行 train 重建与离线仿真准备阶段..."
CUDA_VISIBLE_DEVICES=$CUDA_DEV "$PYTHON_BIN" -u train/run_pipeline.py \
    --train_cfg ./configs/train.yaml \
    --source_path "$PRE_TRAIN_DIR" \
    --model_path "$TRAIN_DIR" \
    --export_path "$EXPORT_DIR" \
    --cuda "$CUDA_DEV" \
    --iteration "$OFFLINE_ITERATION"

echo "========================================================="
echo " 运行成功！全部流程已完成。"
echo " 预处理产物位于  : $PRE_TRAIN_DIR"
echo " 场景重建产物位于: $TRAIN_DIR"
echo " 离线导出产物位于: $EXPORT_DIR"
echo "========================================================="
