import os
import json
from typing import NamedTuple

import numpy as np
import torch
from imageio.v2 import imread
from plyfile import PlyData, PlyElement

from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    K: np.array
    c2w: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic2d: np.array
    optical_image: np.array
    depth: torch.tensor
    mask: np.array
    timestamp: int
    dynamics: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    verts: dict

def getNerfppNorm(cam_info):
    return {'radius': 10}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if 'red' in vertices:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        print('Create random colors')
        shs = np.ones((positions.shape[0], 3)) * 0.5
        colors = SH2RGB(shs)
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def strip_relative_prefix(path):
    return path.replace("\\", "/").removeprefix("./")


def abs_data_path(scene_path, relative_path):
    return os.path.join(scene_path, strip_relative_prefix(relative_path))


def frame_camera_name(frame):
    return frame["camera_name"]


def camera_params(camera_paras, camera_name):
    params = camera_paras["cameras"][camera_name]

    camera_to_ego = np.eye(4, dtype=np.float64)
    camera_to_ego[:3, :3] = np.asarray(params["rotation_matrix"], dtype=np.float64)
    camera_to_ego[:3, 3] = np.asarray(params["translation"], dtype=np.float64)

    intrinsics = params["intrinsics"]
    return {
        "camera_to_ego": camera_to_ego,
        "intrinsic": np.asarray(intrinsics["matrix"], dtype=np.float64),
        "width": int(intrinsics["width"]),
        "height": int(intrinsics["height"]),
    }


def frame_camera_params(camera_paras, frame):
    return camera_params(camera_paras, frame_camera_name(frame))


def frame_camtoworld(camera_paras, frame):
    params = frame_camera_params(camera_paras, frame)
    ego_to_world = np.asarray(frame["ego_to_world"], dtype=np.float64)
    return ego_to_world @ params["camera_to_ego"]


def frame_object_to_ego(dynamic_pose):
    if isinstance(dynamic_pose, dict) and "object_to_ego" in dynamic_pose:
        dynamic_pose = dynamic_pose["object_to_ego"]
    return np.asarray(dynamic_pose, dtype=np.float64)


def frame_object_to_world(frame, dynamic_pose):
    ego_to_world = np.asarray(frame["ego_to_world"], dtype=np.float64)
    return ego_to_world @ frame_object_to_ego(dynamic_pose)


def frame_group_key(frame):
    return os.path.splitext(os.path.basename(strip_relative_prefix(frame["rgb_path"])))[0]


def split_train_test(frames):
    groups, group_order = {}, []
    for idx, frame in enumerate(frames):
        key = frame_group_key(frame)
        if key not in groups:
            groups[key] = len(group_order)
            group_order.append(key)
        yield idx, groups[key]


def readHUGSIMCameras(path, ignore_dynamic):
    train_cam_infos, test_cam_infos = [], []
    with open(os.path.join(path, 'meta_data.json')) as json_file:
        meta_data = json.load(json_file)
    with open(os.path.join(path, 'camera_paras.json')) as json_file:
        camera_paras = json.load(json_file)

    verts = {}
    if 'verts' in meta_data and not ignore_dynamic:
        verts_list = meta_data['verts']
        for k, v in verts_list.items():
            verts[k] = np.array(v)

    frames = meta_data['frames']
    for idx, group_idx in split_train_test(frames):
        frame = frames[idx]
        params = frame_camera_params(camera_paras, frame)
        c2w = frame_camtoworld(camera_paras, frame)
        intrinsic = params["intrinsic"]

        rgb_path = abs_data_path(path, frame['rgb_path'])

        rgb_split = rgb_path.split('/')
        image_name = '_'.join([rgb_split[-2], rgb_split[-1][:-4]])
        image = imread(rgb_path)
        height, width = int(params["height"]), int(params["width"])
        if image.shape[0] != height or image.shape[1] != width:
            raise ValueError(
                f"Image size mismatch for {rgb_path}: image {image.shape[1]}x{image.shape[0]}, "
                f"camera_paras {width}x{height}"
            )

        semantic_2d = None
        semantic_pth = rgb_path.replace("images", "semantics").replace('.png', '.npy').replace('.jpg', '.npy')
        if os.path.exists(semantic_pth):
            semantic_2d = np.load(semantic_pth)
            semantic_2d[(semantic_2d == 14) | (semantic_2d == 15)] = 13

        optical_path = rgb_path.replace("images", "flow").replace('.png', '_flow.npy').replace('.jpg', '_flow.npy')
        if os.path.exists(optical_path):
            optical_image = np.load(optical_path)
        else:
            optical_image = None

        depth_path = rgb_path.replace("images", "depth").replace('.png', '.pt').replace('.jpg', '.pt')
        if os.path.exists(depth_path):
            depth = torch.load(depth_path, weights_only=True)
        else:
            depth = None

        mask = None
        mask_path = rgb_path.replace("images", "masks").replace('.png', '.npy').replace('.jpg', '.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path)

        timestamp = frame.get('timestamp', -1)
        
        dynamics = {}
        if not ignore_dynamic:
            for iid, pose in frame.get('dynamics', {}).items():
                dynamics[iid] = torch.tensor(frame_object_to_world(frame, pose)).cuda()
            
        cam_info = CameraInfo(K=intrinsic, c2w=c2w, image=np.array(image),
                            image_path=rgb_path, image_name=image_name, height=height,
                            width=width, semantic2d=semantic_2d, 
                            optical_image=optical_image, depth=depth, mask=mask, timestamp=timestamp, dynamics=dynamics)
        
        if group_idx % 5 == 4:
            test_cam_infos.append(cam_info)
        else:
            train_cam_infos.append(cam_info)

    return train_cam_infos, test_cam_infos, verts


def readHUGSIMInfo(path, ignore_dynamic):
    train_cam_infos, test_cam_infos, verts = readHUGSIMCameras(path, ignore_dynamic)

    print(f'Loaded {len(train_cam_infos)} train cameras and {len(test_cam_infos)} test cameras')
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        assert False, "Requires for initialize 3d points as inputs"
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print('When loading point clound, meet error:', e)
        exit(0)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           verts=verts)
    return scene_info


sceneLoadTypeCallbacks = {
    "HUGSIM": readHUGSIMInfo,
}
