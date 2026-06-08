import os
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.obj_model import ObjModel
from scene.cameras import cameraList_from_camInfos
import torch
import open3d as o3d
import numpy as np
import shutil


VEHICLE_TEMPLATE = "utils/vehicle_template/benz_nuscenes.ply"


def downsample_points(points, max_points):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices]


def copy_if_different(src, dst):
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)


def copy_if_exists(src, dst):
    if os.path.exists(src):
        copy_if_different(src, dst)


def load_cameras(args, ignore_dynamic=False):
    train_cameras = {}
    test_cameras = {}
    if os.path.exists(os.path.join(args.source_path, "meta_data.json")):
        print("Found meta_data.json file, assuming HUGSIM format data set!")
        scene_info = sceneLoadTypeCallbacks['HUGSIM'](args.source_path, ignore_dynamic)
    else:
        assert False, "Could not recognize scene type! "+args.source_path

    print("Loading Training Cameras")
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
    print("Loading Test Cameras")
    test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)
    return train_cameras, test_cameras, scene_info 

class Scene:

    def __init__(self, args, gaussians:GaussianModel, load_iteration=None, shuffle=True, 
                 ignore_dynamic=False, planning=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "ckpts"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras, self.test_cameras, scene_info = load_cameras(args, ignore_dynamic)

        self.dynamic_verts = scene_info.verts
        self.dynamic_gaussians = {}
        for track_id in scene_info.verts:
            self.dynamic_gaussians[track_id] = ObjModel(args.model.sh_degree, feat_mutable=False)
        if planning is not None:
            for plan_id in planning.keys():
                self.dynamic_gaussians[plan_id] = ObjModel(args.model.sh_degree, feat_mutable=False)

        if not self.loaded_iter:
            copy_if_different(scene_info.ply_path, os.path.join(self.model_path, "input.ply"))
            copy_if_different(os.path.join(args.source_path, 'meta_data.json'), os.path.join(self.model_path, 'meta_data.json'))
            copy_if_different(os.path.join(args.source_path, 'camera_paras.json'), os.path.join(self.model_path, 'camera_paras.json'))
            copy_if_different(os.path.join(args.source_path, 'ground_param.pkl'), os.path.join(self.model_path, 'ground_param.pkl'))
            copy_if_exists(os.path.join(args.source_path, 'geo_reference.json'), os.path.join(self.model_path, 'geo_reference.json'))

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if self.loaded_iter:
            # (model_params, first_iter) = torch.load(os.path.join(self.model_path, "ckpts", f"chkpnt{self.loaded_iter}.pth"))
            (model_params, first_iter) = torch.load(os.path.join(self.model_path, f"scene.pth"))
            gaussians.restore(model_params, None)
            for iid, dynamic_gaussian in self.dynamic_gaussians.items():
                if planning is None or iid not in planning:
                    # (model_params, first_iter) = torch.load(os.path.join(self.model_path, "ckpts", f"dynamic_{iid}_chkpnt{self.loaded_iter}.pth"))
                    (model_params, first_iter) = torch.load(os.path.join(self.model_path, f"dynamic_{iid}.pth"))
                    dynamic_gaussian.restore(model_params, None)
                else:
                    (model_params, first_iter) = torch.load(planning[iid])
                    model_params = list(model_params)
                    model_params.append(None)
                    dynamic_gaussian.restore(model_params, None)
            # for iid, unicycle_pkg in self.unicycles.items():
            #     model_params = torch.load(os.path.join(self.model_path, "ckpts", f"unicycle_{iid}_chkpnt{self.loaded_iter}.pth"))
            #     unicycle_pkg['model'].restore(model_params)

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            dynamic_template_max_points = int(getattr(args, "dynamic_template_max_points", 0) or 0)
            if dynamic_template_max_points > 0:
                print(f"Dynamic template max points: {dynamic_template_max_points}")
            for track_id in self.dynamic_gaussians.keys():
                vertices = scene_info.verts[track_id]

                # init from template
                l, h, w = vertices[:, 0].max() - vertices[:, 0].min(), vertices[:, 1].max() - vertices[:, 1].min(), vertices[:, 2].max() - vertices[:, 2].min()
                pcd = o3d.io.read_point_cloud(VEHICLE_TEMPLATE)
                points = downsample_points(np.asarray(pcd.points), dynamic_template_max_points) * np.array([l, h, w])
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) * 0.5)

                self.dynamic_gaussians[track_id].create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration):
        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        point_cloud_vis_path = os.path.join(self.model_path, "point_cloud_vis/iteration_{}".format(iteration))
        self.gaussians.save_vis_ply(os.path.join(point_cloud_vis_path, "point.ply"))
        for iid, dynamic_gaussian in self.dynamic_gaussians.items():
            dynamic_gaussian.save_vis_ply(os.path.join(point_cloud_vis_path, f"dynamic_{iid}.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
