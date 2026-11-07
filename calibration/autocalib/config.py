"""Recipes and constants.

``dense``           Bridge settings: consecutive frames in batches of 3, 5-frame tracking window, lr 1e-3, 20 epochs.
``dense_robomind``  RobotArena∞ RoboMIND settings: 20 isolated frame pairs, lr 4e-3, 16 epochs.

Both use loss = 1000 x flow L1 + 0.1 x RGB MSE + DINOv2 ViT-S/14 feature cosine. The released Bridge and RoboMIND
poses were fitted with ``dense`` and the overrides in ``tools/sweep_calib.py``.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Recipe:
    name: str
    #: CoTracker window (frames per tracking call); RoboMIND tracks each pair separately (n_frame=1)
    cotracker_n_frame: int
    #: |flow| threshold (px) for the motion mask
    motion_threshold: float
    #: motion-pixel count window for a frame pair to count as "moving" (min, max); max None -> no upper bound
    motion_count: Tuple[int, Optional[int]]
    #: skip the batch when fewer than this many pairs move (None -> never skip)
    skip_if_moving_pairs_leq: Optional[int]
    #: loss terms
    use_rgb_loss: bool
    use_feature_loss: bool
    #: Adam steps per batch and the StepLR(10, 0.5) schedule inside a batch
    steps_per_batch: int
    #: camera learning rate at epoch 0 (halved every 4 epochs)
    camera_lr: float
    epochs: int
    #: frames per optimisation batch (dense recipe): consecutive frames [b*3, b*3+1, b*3+2]
    batch_size: int = 3


DENSE = Recipe(name='dense', cotracker_n_frame=5,
               motion_threshold=1.4, motion_count=(10000, 500000), skip_if_moving_pairs_leq=1,
               use_rgb_loss=True, use_feature_loss=True, steps_per_batch=10, camera_lr=0.001, epochs=20)

#: RoboMIND: same losses, but each (idx, idx+1) pair is one batch, tracked alone, higher lr, 16 epochs
DENSE_ROBOMIND = Recipe(name='dense_robomind', cotracker_n_frame=1,
                        motion_threshold=1.4, motion_count=(10000, 500000), skip_if_moving_pairs_leq=0,
                        use_rgb_loss=True, use_feature_loss=True, steps_per_batch=10, camera_lr=0.004, epochs=16)

RECIPES = {'dense': DENSE, 'dense_robomind': DENSE_ROBOMIND}

#: DINOv2 backbones (torch.hub facebookresearch/dinov2): ViT-S/14 in the optimisation, ViT-B/14 for start-pose scoring
DINO_OPTIMIZE = 'dinov2_vits14'
DINO_SCORE = 'dinov2_vitb14'
DINO_LONGEST_EDGE = 800

#: at most this many frames of a trajectory are used
MAX_FRAMES = 200

# Default start poses (world-to-camera). They only fit the rigs they were set up for; for a new mount make a
# start pose with tools/align_gui.py.
#: BridgeData V1 toy-kitchen over-the-shoulder camera (images0)
BRIDGE_INIT_W2C = np.array([
    [0.40230105648431419, -0.54737938074000059, 0.73384581043452057, 0.0],
    [-0.91514565098874134, -0.21790723695396816, 0.33915317123606575, 0.0],
    [-0.025735139945173187, -0.8080174810137345, -0.58859617136728104, 0.0],
    [-0.1495516053826326, 0.24801468373558327, 0.13542126749117278, 1.0]]).T

#: RoboMIND Franka FR3 cameras (camera_top is not the UR5e mount)
ROBOMIND_INIT_W2C = {
    'camera_left': np.array([
        [-0.81138510327202173, -0.18410696539124252, 0.55476016392913308, 0.0],
        [-0.57686105632703244, 0.40528076144758174, -0.70921000140559831, 0.0],
        [-0.094263120474080767, -0.89546196440434933, -0.43504291101406634, 0.0],
        [0.59219642021433283, 0.52541880645681949, 0.74251763181186836, 1.0]]).T,
    'camera_right': np.array([
        [0.83734427822375257, -0.27765933958399935, 0.47091384654490714, 0.0],
        [-0.54666720000213354, -0.42041948190580292, 0.72415635858278826, 0.0],
        [-0.003087420892972168, -0.86380133733158437, -0.50382310135244501, 0.0],
        [-0.54788088926660228, 0.67045439724472256, 0.68404495102964147, 1.0]]).T,
    'camera_top': np.array([
        [-0.25389173788236485, 0.86150880238723326, -0.43970623016326232, 0.0],
        [0.96252137951360806, 0.18022378814692969, -0.2026622317186689, 0.0],
        [-0.095349774065245474, -0.47468091345164271, -0.87497797171724323, 0.0],
        [0.2258082244284767, -0.54554129322083722, 1.5017372929195465, 1.0]]).T,
}

#: WidowX-250s (Bridge): gripper in [0, 1] (1 = open) -> left/right finger prismatic joints (m)
def bridge_fingers(gripper):
    f = np.asarray(gripper).reshape(-1, 1) * 0.022 + 0.015
    return np.concatenate([f, -f], axis=1)

#: Robotiq 2F-85: gripper closedness in [0, 1] -> 8 mechanism joints (rad), linear map ('legacy' grip mode)
ROBOTIQ_2F85_FINGER_GAINS = np.array([[0.75, -0.4, 0.6, -0.26, 0.75, -0.4, 0.6, -0.26]])

def robotiq_fingers(gripper):
    return np.asarray(gripper).reshape(-1, 1) * ROBOTIQ_2F85_FINGER_GAINS
