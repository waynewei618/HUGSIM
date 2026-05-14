import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose original and rendered front-view videos side by side."
    )
    parser.add_argument("original_video", help="Original collected front-view video.")
    parser.add_argument("rendered_video", help="Rendered front-view video.")
    parser.add_argument("output_video", help="Side-by-side comparison video.")
    return parser.parse_args()


def get_fps(reader):
    meta = reader.get_meta_data()
    fps = meta.get("fps", 10.0)
    if fps is None or fps <= 0:
        return 10.0
    return float(fps)


def as_rgb(image):
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return image[..., :3]


def resize_to_height(image, height):
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def compose_compare_video(original_video, rendered_video, output_video):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    original_reader = imageio.get_reader(original_video)
    rendered_reader = imageio.get_reader(rendered_video)
    fps = get_fps(rendered_reader) or get_fps(original_reader)

    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    frame_count = 0
    try:
        for original_frame, rendered_frame in tqdm(
            zip(original_reader, rendered_reader),
            desc="Composing compare video",
        ):
            original_frame = as_rgb(original_frame)
            rendered_frame = as_rgb(rendered_frame)
            target_height = min(original_frame.shape[0], rendered_frame.shape[0])
            original_frame = resize_to_height(original_frame, target_height)
            rendered_frame = resize_to_height(rendered_frame, target_height)
            writer.append_data(np.concatenate([original_frame, rendered_frame], axis=1))
            frame_count += 1
    finally:
        writer.close()
        original_reader.close()
        rendered_reader.close()

    print(f"Saved {frame_count} comparison frames to {output_video}")


def main():
    args = parse_args()
    compose_compare_video(
        Path(args.original_video).resolve(),
        Path(args.rendered_video).resolve(),
        Path(args.output_video).resolve(),
    )


if __name__ == "__main__":
    main()
