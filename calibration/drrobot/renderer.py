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
"""gsplat rendering of the posed robot (RGB) and of the 2-D motion of its Gaussians between two frames (flow).

Cameras are ``Camera_Pose`` objects; the principal point is the image centre.
"""
import math

import torch
from gsplat import rasterization

from .lbs import lrs


def _lbs_cached(pc, viewpoint_camera, joints_row, means3D, rotations, pose_normalized):
    """Posed means and rotations for one joint vector, memoised in ``pc.lbs_cache`` when it exists.
    The deformation does not depend on the camera, so the cached, detached result is exact."""
    cache = getattr(pc, 'lbs_cache', None)
    if cache is None:
        return lrs(joints_row[None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=pose_normalized,
                   lrs_model=pc.lrs_model, rotations=rotations)
    key = (pose_normalized, tuple(viewpoint_camera.joint_pose.detach().cpu().flatten().tolist()))
    if key not in cache:
        with torch.no_grad():
            m, r = lrs(joints_row[None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=pose_normalized,
                       lrs_model=pc.lrs_model, rotations=rotations)
        cache[key] = (m.detach(), r.detach())
    return cache[key]


def _posed(pc, viewpoint_camera):
    means3D = pc.get_xyz
    rotations = pc.get_rotation
    joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0], 1)
    m, r = _lbs_cached(pc, viewpoint_camera, joints[0], means3D, rotations, False)
    return m[0], r[0]


def render_gradio(viewpoint_camera, pc, bg_color, scaling_modifier=1.0, render_features=False):
    """RGB (and depth) of the robot at ``viewpoint_camera.joint_pose`` (raw joint values)."""
    tanfovx = torch.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = torch.tan(viewpoint_camera.FoVy * 0.5)
    means3D = pc.get_xyz
    focal_length_x = (viewpoint_camera.image_width / (2 * tanfovx)).to(means3D.device)
    focal_length_y = (viewpoint_camera.image_height / (2 * tanfovy)).to(means3D.device)
    # the intrinsics are detached: only the pose receives gradients
    _z = torch.zeros((), device=means3D.device, dtype=torch.float32)
    _o = torch.ones((), device=means3D.device, dtype=torch.float32)
    K = torch.stack([
        torch.stack([focal_length_x.detach().float().reshape(()), _z, _o * (viewpoint_camera.image_width / 2.0)]),
        torch.stack([_z, focal_length_y.detach().float().reshape(()), _o * (viewpoint_camera.image_height / 2.0)]),
        torch.stack([_z, _z, _o]),
    ])

    means3D, rotations = _posed(pc, viewpoint_camera)
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    render_colors, _, _ = rasterization(
        means=means3D,
        quats=rotations,
        scales=pc.get_scaling * scaling_modifier,
        opacities=pc.get_opacity.squeeze(-1),
        colors=pc.get_features,
        viewmats=viewmat[None],
        Ks=K[None],
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=pc.active_sh_degree,
        render_mode='RGB+D',
    )
    return {"render": render_colors[0, :, :, :3].permute(2, 0, 1), "depth": render_colors[0, :, :, 3]}


def point_cloud_to_image(world_coordinates, viewpoint_camera, K):
    means3D_homo = torch.cat([world_coordinates, torch.ones_like(world_coordinates[:, :1])], dim=1)
    means_clip = (viewpoint_camera.full_proj_transform.T @ means3D_homo.T).T
    depth = means_clip[:, 2]
    means_screen = means_clip / (means_clip[:, 3:] + 1e-6)
    x_pixel = (means_screen[:, 0] * 0.5 + 0.5) * int(viewpoint_camera.image_width)
    y_pixel = (means_screen[:, 1] * 0.5 + 0.5) * int(viewpoint_camera.image_height)
    return torch.stack([x_pixel, y_pixel], axis=1), depth


def render_flow(viewpoint_cameras, pc, bg_color, scaling_modifier=1.0, render_features=False):
    """Image of the 2-D displacement of every Gaussian from frame ``viewpoint_cameras[1]`` to
    ``viewpoint_cameras[0]``, splatted at its position in the second camera. Returns {"flow": (1, H, W, 2)}."""
    cam_curr, cam_prev = viewpoint_cameras
    tanfovx = math.tan(cam_prev.FoVx * 0.5)
    tanfovy = math.tan(cam_prev.FoVy * 0.5)
    focal_length_x = cam_prev.image_width / (2 * tanfovx)
    focal_length_y = cam_prev.image_height / (2 * tanfovy)
    K = torch.tensor([[focal_length_x, 0, cam_prev.image_width / 2.0],
                      [0, focal_length_y, cam_prev.image_height / 2.0],
                      [0, 0, 1]], device="cuda")

    means_curr, _ = _posed(pc, cam_curr)
    means_prev, rotations_prev = _posed(pc, cam_prev)
    viewmat = cam_prev.world_view_transform.transpose(0, 1).to(means_prev.device)
    pos_curr, _ = point_cloud_to_image(means_curr, cam_curr, K)
    pos_prev, _ = point_cloud_to_image(means_prev, cam_prev, K)
    flow_2d = pos_curr - pos_prev

    rendered_flow, _, _ = rasterization(
        means=means_prev,
        quats=rotations_prev,
        scales=pc.get_scaling * scaling_modifier,
        opacities=pc.get_opacity.squeeze(-1),
        colors=flow_2d,
        viewmats=viewmat[None],
        Ks=K[None],
        backgrounds=torch.zeros((1, 2)).cuda(),
        width=int(cam_prev.image_width),
        height=int(cam_prev.image_height),
        packed=False,
        sh_degree=None,
    )
    return {"flow": rendered_flow}
