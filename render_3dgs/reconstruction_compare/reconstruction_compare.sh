#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <scene_path> <original_image_path>"
  echo "Example: $0 /workspace/HUGSIM/outputs/nusc/scene-0038 /workspace/data/HUGSIM/nusc/scene-0038"
  exit 1
fi

SCENE_PATH="$1"
ORIGINAL_IMAGE_PATH="$2"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"

if [[ ! -d "$SCENE_PATH" ]]; then
  echo "Scene path does not exist: $SCENE_PATH" >&2
  exit 1
fi

if [[ ! -e "$ORIGINAL_IMAGE_PATH" ]]; then
  echo "Original image path does not exist: $ORIGINAL_IMAGE_PATH" >&2
  exit 1
fi

if [[ ! -f "$SCENE_PATH/meta_data.json" ]]; then
  echo "Scene path must contain meta_data.json: $SCENE_PATH" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENE_ABS="$(realpath "$SCENE_PATH")"
mkdir -p "$REPO_ROOT/outputs"
OUTPUTS_ABS="$(realpath "$REPO_ROOT/outputs")"
OUTPUT_REL="$(basename "$SCENE_ABS")"
if [[ "$SCENE_ABS" == "$OUTPUTS_ABS/"* ]]; then
  OUTPUT_REL="${SCENE_ABS#"$OUTPUTS_ABS"/}"
fi
OUTPUT_PATH="$REPO_ROOT/outputs/render_3dgs/$OUTPUT_REL"
mkdir -p "$OUTPUT_PATH"

echo "Scene: $SCENE_PATH"
echo "Original images: $ORIGINAL_IMAGE_PATH"
echo "Output: $OUTPUT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

pixi run python render_3dgs/reconstruction_compare/extract_scene_inputs.py \
  "$SCENE_PATH" \
  "$ORIGINAL_IMAGE_PATH" \
  "$OUTPUT_PATH"

if [[ ! -f "$OUTPUT_PATH/aeb_front_original.mp4" ]]; then
  echo "Failed to create original front video from: $ORIGINAL_IMAGE_PATH" >&2
  exit 1
fi

pixi run python render_3dgs/reconstruction_compare/compose_compare_video.py \
  "$SCENE_PATH" \
  "$OUTPUT_PATH/aeb_front_original.mp4" \
  "$OUTPUT_PATH/aeb_trajectory.json" \
  "$OUTPUT_PATH/aeb_camera.json" \
  "$OUTPUT_PATH/aeb_front_compare.mp4" \
  --real-camera-output "$OUTPUT_PATH/aeb_real_front_120_rendered.mp4"

echo "Comparison video: $OUTPUT_PATH/aeb_front_compare.mp4"
echo "Real vehicle AEB front render: $OUTPUT_PATH/aeb_real_front_120_rendered.mp4"
echo "Real vehicle AEB render timing: $OUTPUT_PATH/aeb_real_front_120_rendered.timing.csv"
