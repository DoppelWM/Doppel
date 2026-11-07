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
"""Trained Dr.Robot Gaussian-splat robot model (inference only).

A model directory contains::

    robot_xml/model.xml (or scene.xml)            MJCF of the robot, defines the kinematic chain
    point_cloud/pose_conditioned_iteration_N/
        point_cloud.ply                           canonical Gaussians
        lrs_model.pth                             implicit skinning-weight MLP
    canonical_joint_positions.npy                 optional: rest pose of the canonical Gaussians
"""
import os

import numpy as np
import torch
from plyfile import PlyData

from . import lbs
from .kinematics import build_chain_from_mjcf_path
from .nn import IRS


class GaussianModel:
    def __init__(self, sh_degree=3, device='cuda'):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.device = device
        self.chain = None
        self.lrs_model = None
        self._lrs = None
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def initialize_kinematic_chain(self, chain):
        self.chain = chain.to(dtype=torch.float, device=self.device)
        num_joints = self.chain.n_joints
        num_frames = len(self.chain.joint_type_indices)
        # input: xyz, per-joint offsets and distances, positional encoding x10 + raw values
        self.lrs_model = IRS((3 + num_joints * 3 + num_joints) * 11, num_frames).to(self.device)

    def load_ply(self, path):
        el = PlyData.read(path).elements[0]
        xyz = np.stack((np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])), axis=1)
        opacities = np.asarray(el["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        for c in range(3):
            features_dc[:, c, 0] = np.asarray(el[f"f_dc_{c}"])
        extra_f_names = sorted([p.name for p in el.properties if p.name.startswith("f_rest_")], key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.stack([np.asarray(el[n]) for n in extra_f_names], axis=1)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = sorted([p.name for p in el.properties if p.name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(el[n]) for n in scale_names], axis=1)
        rot_names = sorted([p.name for p in el.properties if p.name.startswith("rot")], key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([np.asarray(el[n]) for n in rot_names], axis=1)

        def param(a):
            return torch.nn.Parameter(torch.tensor(a, dtype=torch.float, device=self.device).requires_grad_(True))

        self._xyz = param(xyz)
        self._features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = param(opacities)
        self._scaling = param(scales)
        self._rotation = param(rots)
        self.active_sh_degree = self.max_sh_degree
        self.lrs_model.load_state_dict(torch.load(os.path.join(os.path.dirname(path), 'lrs_model.pth'), map_location=self.device))


def latest_iteration(model_path):
    its = [int(d.split('_')[-1]) for d in os.listdir(os.path.join(model_path, 'point_cloud')) if d.startswith('pose_conditioned_iteration_')]
    if not its:
        raise FileNotFoundError(f'no point_cloud/pose_conditioned_iteration_* in {model_path}')
    return max(its)


def load_gaussians(model_path, iteration=None, device='cuda'):
    """Load a model directory (see module docstring); ``iteration=None`` takes the highest saved iteration."""
    model_path = os.path.abspath(model_path)
    xml = os.path.join(model_path, 'robot_xml', 'model.xml')
    if not os.path.exists(xml):
        xml = os.path.join(model_path, 'robot_xml', 'scene.xml')
    cp = os.path.join(model_path, 'canonical_joint_positions.npy')
    lbs.CANONICAL_POSE_OVERRIDE = np.load(cp) if os.path.exists(cp) else None
    g = GaussianModel(device=device)
    g.initialize_kinematic_chain(build_chain_from_mjcf_path(xml))
    it = iteration or latest_iteration(model_path)
    g.load_ply(os.path.join(model_path, 'point_cloud', f'pose_conditioned_iteration_{it}', 'point_cloud.ply'))
    g.model_path, g.iteration = model_path, it
    return g
