import numpy as np


def as_camera_intrinsics(value, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (3, 3):
        output = np.eye(4, dtype=np.float32)
        output[:3, :3] = matrix
        return output
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"{name} must be a real 3x3 or 4x4 camera intrinsic matrix")


def as_transform(value, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a real 4x4 transform matrix")
    return matrix


def as_positive_int(value, name):
    output = int(value)
    if output <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return output


def as_positive_float(value, name):
    output = float(value)
    if not np.isfinite(output) or output <= 0.0:
        raise ValueError(f"{name} must be a finite value greater than 0")
    return output


def render_to_uint8(render_tensor):
    image = render_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)
