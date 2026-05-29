from torch.utils.data import Dataset

class HUGSIM_dataset(Dataset):
    def __init__(self, views):
        super().__init__()
        self.views = views
        self.gap = self._infer_camera_gap()

    def _infer_camera_gap(self):
        if not self.views:
            return 1
        first_frame = self.views[0].image_name.rsplit("_", 1)[-1]
        return max(1, sum(view.image_name.rsplit("_", 1)[-1] == first_frame for view in self.views))
    
    def __getitem__(self, index):
        if index - self.gap >= 0:
            prev_index = index-self.gap
        else:
            prev_index = -1

        viewpoint_cam = self.views[index]

        gt_image = viewpoint_cam.original_image
        if viewpoint_cam.semantic2d is not None:
            gt_semantic = viewpoint_cam.semantic2d
        else:
            gt_semantic = None
        if viewpoint_cam.optical_gt is not None:
            gt_optical = viewpoint_cam.optical_gt
        else:
            gt_optical = None
        if viewpoint_cam.depth is not None:
            gt_depth = viewpoint_cam.depth
        else:
            gt_depth = None
        if viewpoint_cam.mask is not None:
            mask = viewpoint_cam.mask
        else:
            mask = None

        return index, prev_index, gt_image, gt_semantic, gt_optical, gt_depth, mask

    def __len__(self):
        return len(self.views)
    
def tocuda(ans):
    if ans is None:
        return None
    else:
        return ans.cuda()
    
def hugsim_collate(data):
    assert len(data) == 1
    return data[0]
