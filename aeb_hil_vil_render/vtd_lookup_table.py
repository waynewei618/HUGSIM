from pathlib import Path

import cv2
import numpy as np


def default_lookup_cache_path(output_video, lookup_table_path, width, height):
    output_video = Path(output_video)
    lookup_table_path = Path(lookup_table_path)
    cache_name = f"{lookup_table_path.stem}_{int(width)}x{int(height)}.npz"
    return output_video.parent / ".lookup_cache" / cache_name


def load_lookup_maps(lookup_table_path, cache_path=None):
    lookup_table_path = Path(lookup_table_path)
    stat = lookup_table_path.stat()

    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = _load_cached_maps(cache_path, lookup_table_path, stat)
        if cached is not None:
            return cached

    map_x, map_y = _parse_lookup_table(lookup_table_path)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            map_x=map_x,
            map_y=map_y,
            source_path=np.asarray(str(lookup_table_path)),
            source_size=np.asarray(stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(stat.st_mtime_ns, dtype=np.int64),
        )
        print(f"Saved lookup table cache to {cache_path}")

    return map_x, map_y


def _load_cached_maps(cache_path, lookup_table_path, stat):
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path) as data:
            source_path = str(data["source_path"])
            source_size = int(data["source_size"])
            source_mtime_ns = int(data["source_mtime_ns"])
            if (
                source_path != str(lookup_table_path)
                or source_size != stat.st_size
                or source_mtime_ns != stat.st_mtime_ns
            ):
                return None
            return data["map_x"].astype(np.float32), data["map_y"].astype(np.float32)
    except Exception as exc:
        print(f"Warning: failed to load lookup table cache {cache_path}: {exc}")
        return None


def _parse_lookup_table(lookup_table_path):
    with lookup_table_path.open("r") as f:
        version = f.readline().strip()
        try:
            width, height = [int(value) for value in f.readline().split()]
        except ValueError as exc:
            raise ValueError(f"{lookup_table_path} has an invalid lookup table size header") from exc

        if version != "1":
            print(f"Warning: lookup table {lookup_table_path} version is {version}, expected 1")

        map_x = np.empty((height, width), dtype=np.float32)
        map_y = np.empty((height, width), dtype=np.float32)
        expected_values = width * 3

        for y in range(height):
            line = f.readline()
            if not line:
                raise ValueError(f"{lookup_table_path} ended before row {y}")
            row = np.fromstring(line.replace("|", ","), dtype=np.float32, sep=",")
            if row.size != expected_values:
                raise ValueError(
                    f"{lookup_table_path} row {y} has {row.size} values; expected {expected_values}"
                )
            row = row.reshape(width, 3)
            map_x[y] = row[:, 0]
            map_y[y] = row[:, 1]

    print(f"Loaded VTD lookup table {lookup_table_path} ({width}x{height})")
    return map_x, map_y


class VtdLookupRemapper:
    def __init__(self, lookup_table_path, cache_path=None):
        self.lookup_table_path = Path(lookup_table_path)
        self.map_x, self.map_y = load_lookup_maps(self.lookup_table_path, cache_path)
        self.height, self.width = self.map_x.shape
        self.invalid_mask = (self.map_x == 0.0) & (self.map_y == 0.0)

    def remap(self, image):
        if image.shape[0] != self.height or image.shape[1] != self.width:
            raise ValueError(
                f"lookup table {self.lookup_table_path} expects {self.width}x{self.height}, "
                f"got image {image.shape[1]}x{image.shape[0]}"
            )

        output = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        output[self.invalid_mask] = 0
        return output
