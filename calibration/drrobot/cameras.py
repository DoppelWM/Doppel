#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from torch import nn

from .graphics import getProjectionMatrix, se3_to_SE3


class Camera_Pose(nn.Module):
    """A camera whose pose is an se(3) twist (w, v) applied on the camera-to-world side of a start pose.

    The principal point is the image centre. ``fx`` is a parameter but receives no gradient through the
    renderer (the intrinsics are rebuilt from detached values), so only the 6-DoF pose is optimised.
    """

    def __init__(self, start_pose_w2c, FoVx, FoVy, image_width, image_height, joint_pose=None, time=0, depth=None,
                 zero_init=False):
        super().__init__()
        FoVx = torch.tensor(FoVx, dtype=torch.float32) if not isinstance(FoVx, torch.Tensor) else FoVx
        FoVy = torch.tensor(FoVy, dtype=torch.float32) if not isinstance(FoVy, torch.Tensor) else FoVy
        tanfovx = torch.tan(FoVx * 0.5)
        fx = image_width / (2 * tanfovx)
        self.fx = nn.Parameter(fx, requires_grad=True)
        self.fy = fx
        self.FoVx = 2 * torch.atan(image_width / (2 * self.fx))
        self.FoVy = 2 * torch.atan(image_height / (2 * self.fy))
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0
        self.znear = 0.01
        self.time = time
        self.depth = depth
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        self.device = start_pose_w2c.device
        self.start_pose_w2c = start_pose_w2c
        self.joint_pose = joint_pose

    def invalidate_cache(self):
        """Call after every optimiser step that changes w / v."""
        self._wvt_cache = None
        self._fpt_cache = None

    @property
    def world_view_transform(self):
        # transposed world-to-camera; memoised until invalidate_cache()
        if getattr(self, '_wvt_cache', None) is None:
            wvt = torch.matmul(se3_to_SE3(self.w, self.v), torch.linalg.inv_ex(self.start_pose_w2c.detach()).inverse)
            self._wvt_cache = torch.linalg.inv_ex(wvt).inverse.transpose(0, 1)
        return self._wvt_cache

    @property
    def full_proj_transform(self):
        if getattr(self, '_fpt_cache', None) is None:
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(self.device)
            self._fpt_cache = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        return self._fpt_cache
