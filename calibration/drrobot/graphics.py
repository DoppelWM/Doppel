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
import math

import torch


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    return torch.stack([torch.stack([O, -w2, w1], dim=-1),
                        torch.stack([w2, O, -w0], dim=-1),
                        torch.stack([-w1, w0, O], dim=-1)], dim=-2)


def taylor_A(x, nth=10):
    # sin(x) / x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        if i > 0:
            denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_B(x, nth=10):
    # (1 - cos(x)) / x^2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_C(x, nth=10):
    # (x - sin(x)) / x^3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def se3_to_SE3(w, v):
    """Exponential map of the twist (w, v) -> 4x4 rigid transform."""
    deltaT = torch.zeros((4, 4), device=w.device)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    deltaT[:3, :3] = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    deltaT[:3, 3] = V @ v
    deltaT[3, 3] = 1.
    return deltaT
