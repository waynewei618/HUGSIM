import numpy as np
import pyquaternion


ALLOWED_RIGID_CLASSES = (
    "Car",
    "Pickup Truck",
    "Medium-sized Truck",
    "Semi-truck",
    "Towed Object",
    "Motorcycle",
    "Other Vehicle - Construction Vehicle",
    "Other Vehicle - Uncommon",
    "Other Vehicle - Pedicab",
    "Emergency Vehicle",
    "Bus",
    "Personal Mobility Device",
    "Motorized Scooter",
    "Bicycle",
    "Train",
    "Trolley",
    "Tram / Subway",
)

ALLOWED_NONRIGID_CLASSES = (
    "Pedestrian",
    "Pedestrian with Object",
)


def _pandaset_pose_to_matrix(pose):
    translation = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]])
    quaternion = np.array([pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]])
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = pyquaternion.Quaternion(quaternion).rotation_matrix
    pose_matrix[:3, 3] = translation
    return pose_matrix


def _yaw_to_rotation_matrix(yaw: np.ndarray):
    rotation_matrices = np.zeros((yaw.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = np.cos(yaw)
    rotation_matrices[:, 0, 1] = -np.sin(yaw)
    rotation_matrices[:, 1, 0] = np.sin(yaw)
    rotation_matrices[:, 1, 1] = np.cos(yaw)
    rotation_matrices[:, 2, 2] = 1
    return rotation_matrices


def get_vertices(dim, bottom_center=None):
    if bottom_center is None:
        bottom_center = np.array([0.0, 0.0, 0.0])

    vertices = bottom_center[None, :].repeat(8, axis=0)
    vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
    vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2
    vertices[[0, 1, 4, 5], 1] = vertices[[0, 1, 4, 5], 1] + dim[1] / 2
    vertices[[2, 3, 6, 7], 1] = vertices[[2, 3, 6, 7], 1] - dim[1] / 2
    vertices[[0, 2, 5, 7], 2] = vertices[[0, 2, 5, 7], 2] + dim[2] / 2
    vertices[[1, 3, 4, 6], 2] = vertices[[1, 3, 4, 6], 2] - dim[2] / 2
    return vertices
