from render_3dgs.core.camera_math import as_positive_int
from render_3dgs.core.camera_view_render import CameraViewRender
from render_3dgs.core.tiled_camera_renderer import (
    DEFAULT_TILE_GUARD_PIXELS,
    DEFAULT_TILE_RENDER_BATCH_SIZE,
    TiledCameraRenderer,
)


class GaussianSceneRenderer:
    def __init__(
        self,
        scene_path,
        near_plane=0.01,
        far_plane=500.0,
        ego_trajectory=None,
        insert_vehicle_id=None,
        insert_vehicle_s=None,
        realcar_path=None,
        insert_vehicle_height=-0.3,
        tile_guard_pixels=DEFAULT_TILE_GUARD_PIXELS,
        tile_render_batch_size=DEFAULT_TILE_RENDER_BATCH_SIZE,
    ):
        self.camera_view_render = CameraViewRender(
            scene_path=scene_path,
            near_plane=near_plane,
            far_plane=far_plane,
            ego_trajectory=ego_trajectory,
            insert_vehicle_id=insert_vehicle_id,
            insert_vehicle_s=insert_vehicle_s,
            realcar_path=realcar_path,
            insert_vehicle_height=insert_vehicle_height,
        )
        self.tile_guard_pixels = as_positive_int(tile_guard_pixels, "tile_guard_pixels")
        self.tile_render_batch_size = as_positive_int(tile_render_batch_size, "tile_render_batch_size")
        self._tiled_renderer = None

    def reset_temporal_state(self):
        self.camera_view_render.reset_temporal_state()

    @property
    def near_plane(self):
        return self.camera_view_render.near_plane

    @property
    def far_plane(self):
        return self.camera_view_render.far_plane

    @property
    def scene_path(self):
        return self.camera_view_render.scene_path

    @property
    def gaussians(self):
        return self.camera_view_render.gaussians

    @property
    def dynamic_gaussians(self):
        return self.camera_view_render.dynamic_gaussians

    @property
    def background(self):
        return self.camera_view_render.background

    @property
    def previous_camera(self):
        return self.camera_view_render.previous_camera

    @previous_camera.setter
    def previous_camera(self, value):
        self.camera_view_render.previous_camera = value

    def render_planes(self, near_plane=None, far_plane=None):
        return self.camera_view_render.render_planes(near_plane, far_plane)

    def _make_camera(self, *args, **kwargs):
        return self.camera_view_render._make_camera(*args, **kwargs)

    def _get_tiled_renderer(self):
        if self._tiled_renderer is None:
            self._tiled_renderer = TiledCameraRenderer(
                self.camera_view_render,
                guard_pixels=self.tile_guard_pixels,
                render_batch_size=self.tile_render_batch_size,
            )
        return self._tiled_renderer

    def render_camera(
        self,
        intrinsics,
        camera_to_world,
        width,
        height,
        timestamp=0.0,
        dynamics=None,
        image_name="camera_render",
        near_plane=None,
        far_plane=None,
        tile_rows=1,
        tile_cols=1,
    ):
        tile_rows = as_positive_int(tile_rows, "tile_rows")
        tile_cols = as_positive_int(tile_cols, "tile_cols")
        if tile_rows == 1 and tile_cols == 1:
            return self.camera_view_render.render_camera(
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                width=width,
                height=height,
                timestamp=timestamp,
                dynamics=dynamics,
                image_name=image_name,
                near_plane=near_plane,
                far_plane=far_plane,
            )
        return self._get_tiled_renderer().render_tiled_camera(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            width=width,
            height=height,
            timestamp=timestamp,
            dynamics=dynamics,
            image_name=image_name,
            near_plane=near_plane,
            far_plane=far_plane,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
