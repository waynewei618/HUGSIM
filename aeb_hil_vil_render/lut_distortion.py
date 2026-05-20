from pathlib import Path

import numpy as np


def read_vtd_lookup_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VTD lookup table not found: {path}")

    with path.open("rb") as f:
        version_line = f.readline().strip()
        if version_line != b"1":
            raise ValueError(f"{path} uses unsupported lookup table version: {version_line!r}")

        size_line = f.readline().strip()
        try:
            width, height = [int(value) for value in size_line.split()]
        except ValueError as exc:
            raise ValueError(f"{path} has invalid lookup table size line: {size_line!r}") from exc

        map_x = np.empty((height, width), dtype=np.float32)
        map_y = np.empty((height, width), dtype=np.float32)
        expected_values = width * 3
        for row in range(height):
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended before row {row}")

            values = np.fromstring(line.replace(b"|", b","), sep=",", dtype=np.float32)
            if values.size != expected_values:
                raise ValueError(
                    f"{path} row {row} contains {values.size} values, expected {expected_values}"
                )
            triples = values.reshape(width, 3)
            map_x[row] = triples[:, 0]
            map_y[row] = triples[:, 1]

    return map_x, map_y


def apply_lookup_table_distortion(image, map_x, map_y):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("lookup-table distortion requires OpenCV; run through the project pixi environment") from exc

    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be HxW or HxWxC, got shape {image.shape}")
    if image.shape[:2] != map_x.shape:
        raise ValueError(
            f"image size {image.shape[1]}x{image.shape[0]} does not match "
            f"lookup table size {map_x.shape[1]}x{map_x.shape[0]}"
        )

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class LookupTableDistorter:
    def __init__(self, lookup_table_path):
        self.lookup_table_path = Path(lookup_table_path)
        self.map_x, self.map_y = read_vtd_lookup_table(self.lookup_table_path)

    def __call__(self, image):
        return apply_lookup_table_distortion(image, self.map_x, self.map_y)
