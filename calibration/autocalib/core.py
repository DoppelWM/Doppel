"""Camera optimisation shared by all datasets.

All cameras of a batch share one set of ``Camera_Pose`` parameters (``self.first_camera``), which persists across
batches and epochs. Every batch builds a new Adam optimiser over the parameter list with one entry per camera,
i.e. the same tensors several times; Adam then applies ``batch_size`` updates per step with shared moments, so the
effective learning rate is about ``batch_size x camera_lr``. The recipes were tuned with this behaviour.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from drrobot.cameras import Camera_Pose
from drrobot.renderer import render_flow, render_gradio

from .config import DINO_LONGEST_EDGE, DINO_OPTIMIZE, DINO_SCORE, Recipe
from .features import cosine_loss, load_dinov2, patch_tokens


def iou_loss(pred, target, smooth=1e-6):
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (1 - (intersection + smooth) / (union + smooth)).mean()


class CameraOptimizer:
    def __init__(self, gaussians, background_color, recipe: Recipe, W, H, intrinsics, device='cuda'):
        self.gaussians = gaussians
        self.background_color = background_color
        self.recipe = recipe
        self.W, self.H = int(W), int(H)
        self.intrinsics = np.asarray(intrinsics)
        self.device = device
        self.first_camera = None
        self.dinov2 = load_dinov2(DINO_OPTIMIZE, device) if recipe.use_feature_loss else None
        # the skinning depends only on the joints, never on the camera: memoise it per joint vector
        self.gaussians.lbs_cache = {}

    def reset(self):
        """Forget the shared camera parameters."""
        self.first_camera = None

    def _fov(self):
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        return 2 * np.arctan(self.W / (2 * fx)), 2 * np.arctan(self.H / (2 * fy))

    def _make_cameras(self, joints, init_w2c):
        fovx, fovy = self._fov()
        cameras = []
        for c_idx in range(len(joints)):
            camera = Camera_Pose(torch.tensor(init_w2c).clone().detach().float().cuda(), fovx, fovy,
                                 self.W, self.H, joint_pose=torch.tensor(joints[c_idx]), zero_init=True).cuda()
            if self.first_camera is None:
                self.first_camera = camera
            else:
                for name, param in self.first_camera.named_parameters():
                    if name in camera._parameters:
                        camera._parameters[name] = param
            cameras.append(camera)
        return cameras

    def _render_batch(self, cameras, with_flow=True):
        """Images (B,3,H,W), soft robot masks (B,H,W) and rendered flows (B-1,2,H,W) of consecutive frames."""
        images, masks, flows = [], [], []
        for i, camera in enumerate(cameras):
            image = render_gradio(camera, self.gaussians, self.background_color)['render']
            if with_flow and i > 0:
                flow = render_flow([cameras[i], cameras[i - 1]], self.gaussians, self.background_color)['flow']
                _, H_flow, W_flow, _ = flow.shape
                flows.append(flow.squeeze().reshape((-1, 2)).reshape((H_flow, W_flow, 2)).permute(2, 0, 1))
            images.append(image)
            masks.append(torch.sigmoid((image - 0.01) * 500).max(dim=0)[0])
        return torch.stack(images), torch.stack(masks), (torch.stack(flows) if flows else None)

    def _feature_loss(self, dinov2, rendered, real01, masks):
        features, gh, gw = patch_tokens(dinov2, rendered, DINO_LONGEST_EDGE)
        real_features, _, _ = patch_tokens(dinov2, real01, DINO_LONGEST_EDGE)
        resized_mask = F.interpolate(masks.unsqueeze(1).float(), size=(gh, gw), mode='nearest')
        resized_mask = resized_mask.reshape(features.shape[0], -1)
        features = features * resized_mask.squeeze().unsqueeze(-1)
        return cosine_loss(features, real_features)

    def current_w2c(self):
        return self.first_camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()

    def optimize_batch(self, images, joints, init_w2c, fwd_flows, fwd_valids, camera_lr, steps, epoch, frame_ids=None):
        """Optimise the shared camera on one batch of consecutive frames.

        images (B,H,W,3) uint8, joints (B,J), init_w2c (4,4) the start pose the twist is applied to,
        fwd_flows (B-1,2,H,W) / fwd_valids (B-1,1,H,W) the real flow of the batch's frame pairs.
        Returns the world-to-camera matrix after the batch.
        """
        r, device = self.recipe, self.device
        real = torch.stack([torch.from_numpy(im).float().cuda().permute(2, 0, 1).to(device, non_blocking=True) for im in images])
        cameras = self._make_cameras(joints, init_w2c)
        optimizer = torch.optim.Adam([p for camera in cameras for p in camera.parameters()], lr=camera_lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        for step in range(steps):
            optimizer.zero_grad()
            rendered, masks, gaussian_flows = self._render_batch(cameras)

            if step == 0:
                fwd_flows = fwd_flows.to(device)
                fwd_valids = (fwd_valids.to(device) > 0.3).float()
                motion_mask = torch.norm(fwd_flows, dim=1, keepdim=True) > r.motion_threshold
                counts = motion_mask.sum(dim=(1, 2, 3))
                lo, hi = r.motion_count
                moving = (counts >= lo) if hi is None else ((counts >= lo) & (counts < hi))
                moving = moving.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
                if r.skip_if_moving_pairs_leq is not None and moving.sum() <= r.skip_if_moving_pairs_leq:
                    break   # too few moving frame pairs: the batch does not constrain the camera

            # flow L1, normalised by the image size, on frame pairs with enough motion
            flow_loss = (gaussian_flows - fwd_flows).abs()
            flow_loss[:, 0] /= self.W
            flow_loss[:, 1] /= self.H
            flow_loss = (flow_loss * fwd_valids * moving).sum() / fwd_valids.sum() * 1000
            rgb_loss = F.mse_loss(real / 255, rendered, reduction='none').mean(dim=(1, 2, 3)).mean()
            total_loss = flow_loss + rgb_loss * 0.1 if r.use_rgb_loss else flow_loss
            feat_loss = None
            if r.use_feature_loss:
                feat_loss = self._feature_loss(self.dinov2, rendered, real / 255, masks)
                total_loss = total_loss + feat_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            for camera in cameras:
                camera.invalidate_cache()
            if step == steps - 1:
                print(f'epoch {epoch} batch {frame_ids}: flow {flow_loss.item():.5f} rgb {rgb_loss.item():.5f}'
                      + (f' feat {feat_loss.item():.5f}' if feat_loss is not None else ''))

        return cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy()

    def _alpha(self, cam):
        """Rendered-robot coverage: one minus the difference of renders on white and on black backgrounds."""
        white, black = torch.ones(3, device=self.device), torch.zeros(3, device=self.device)
        r_w = render_gradio(cam, self.gaussians, white)['render'].clamp(0, 1)
        r_b = render_gradio(cam, self.gaussians, black)['render'].clamp(0, 1)
        return 1 - (r_w - r_b).abs().max(0)[0]

    @torch.no_grad()
    def motion_overlap(self, joints, w2c, fwd_flows, fwd_valids, pair_ids, threshold=0.15):
        """(recall, precision) of the real motion mask (|flow| > motion_threshold on valid pixels of pair (i, i+1))
        by the rendered robot at frame i, averaged over the pairs in ``pair_ids`` that move."""
        fovx, fovy = self._fov()
        rec, prec = [], []
        for i in pair_ids:
            cam = Camera_Pose(torch.tensor(w2c).clone().detach().float().cuda(), fovx, fovy, self.W, self.H,
                              joint_pose=torch.tensor(joints[i]), zero_init=True).cuda()
            alpha = (self._alpha(cam) > threshold).float()
            valid = (fwd_valids[i, 0].to(self.device) > 0.3).float()
            motion = (torch.norm(fwd_flows[i].to(self.device), dim=0) > self.recipe.motion_threshold).float() * valid
            if motion.sum() < 100:
                continue
            rec.append(float((alpha * motion).sum() / motion.sum()))
            prec.append(float((alpha * motion).sum() / alpha.sum().clamp(min=1)))
        return (float(np.mean(rec)) if rec else 0.0, float(np.mean(prec)) if prec else 0.0)

    def robot_pixels(self, joints, w2c, threshold=0.15):
        """Pixels covered by the rendered robot at ``w2c`` for each joint vector."""
        cams = CameraOptimizer(self.gaussians, self.background_color, self.recipe, self.W, self.H, self.intrinsics,
                               self.device)._make_cameras(joints, w2c)
        return [int((self._alpha(cam) > threshold).sum().item()) for cam in cams]

    def score_pose(self, images, joints, init_w2c, fwd_flows, fwd_valids, mode='feature'):
        """Score a candidate start pose without optimising (lower is better).

        'feature'   DINOv2 ViT-B/14 masked feature loss (Bridge default)
        'flow'      1000 x masked flow L1 (inf if exactly zero)
        'alpha_iou' IoU loss of the rendered-robot coverage vs the real motion mask
        Returns (w2c, score).
        """
        device = self.device
        cameras = self._make_cameras(joints, init_w2c)
        w2c = cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy()
        if mode == 'alpha_iou':
            with torch.no_grad():
                alphas = torch.stack([self._alpha(cam).clamp(0, 1) for cam in cameras[:-1]])
            motion = (torch.norm(fwd_flows.to(device), dim=1) > self.recipe.motion_threshold).float()
            return w2c, iou_loss(alphas, motion).item()

        real = torch.stack([torch.from_numpy(im).float().cuda().permute(2, 0, 1).to(device, non_blocking=True) for im in images])
        rendered, masks, gaussian_flows = self._render_batch(cameras)
        if mode == 'feature':
            return w2c, self._feature_loss(load_dinov2(DINO_SCORE, device), rendered, real / 255, masks).item()
        fwd_flows = fwd_flows.to(device)
        fwd_valids = (fwd_valids.to(device) > 0.3).float()
        moving = (torch.norm(fwd_flows, dim=1, keepdim=True).gt(self.recipe.motion_threshold).sum(dim=(1, 2, 3)) > 10000).float()
        moving = moving.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        flow_loss = (gaussian_flows - fwd_flows).abs()
        flow_loss[:, 0] /= self.W
        flow_loss[:, 1] /= self.H
        flow_loss = (flow_loss * fwd_valids * moving).sum() / fwd_valids.sum() * 1000
        if flow_loss == 0:
            return None, float('inf')
        return w2c, flow_loss.item()
