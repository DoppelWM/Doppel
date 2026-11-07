"""One gripper mapping for every render model: logged gripper value -> finger-joint qpos -> normalised pose.

The Gaussian robot models are driven with the full MuJoCo qpos in chain order (arm joints followed by the
gripper mechanism joints), normalised to [-1, 1] with the chain's joint limits (``lrs(..., pose_normalized=True)``).
Datasets log one scalar per frame for the gripper. This module turns that scalar into the mechanism joints:

* **Robotiq 2F-85** (Franka FR3 / UR5e models, 8 mechanism joints, tendon actuator with ctrl in [0, 255]):
  a settled look-up table. For 33 closure values the actuator is commanded, the arm is held still and the
  physics is stepped until the mechanism rests; the resting qpos of the eight joints is recorded. Contacts stay
  on, so the coupler / spring-link / follower angles are the ones the real mechanism reaches.
* **WidowX-250s** (Trossen model, two prismatic finger joints tied by an equality): a linear map derived from the
  actuator's ctrlrange -- open = upper bound, closed = lower bound, right finger = -left finger.

``closure_from_log`` converts a dataset's gripper log to closure in [0, 1] (0 = open, 1 = closed), the logged
value itself (clipped), continuous in time:

    convention 'closure'  -- the log is closure already (0 = open)
    convention 'openness' -- the log is openness (1 = open) -> closure = 1 - g
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

FINGER_JOINT_RE = re.compile(r"(driver|coupler|spring_link|follower|finger)")
ACTUATOR_CANDIDATES = ("fingers_actuator", "robotiq_fingers_actuator", "robotiq_2f85/fingers_actuator", "gripper")


def resolve_model_xml(model_xml_dir: str) -> str:
    """``robot_xml/`` dir (or a file) -> the MJCF to load. Prefer scene.xml (what the renderer's chain uses)."""
    # Always absolute: with a relative path MuJoCo resolves the included file's meshdir twice for the UR5e model.
    model_xml_dir = os.path.abspath(os.path.expanduser(model_xml_dir))
    if os.path.isfile(model_xml_dir):
        return model_xml_dir
    for name in ("scene.xml", "model.xml"):
        p = os.path.join(model_xml_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no scene.xml/model.xml under {model_xml_dir}")


def closure_from_log(g: np.ndarray, convention: str = "closure") -> np.ndarray:
    """Dataset gripper log -> closure in [0, 1]. See the module docstring."""
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if convention == "closure":
        c = g
    elif convention == "openness":
        c = 1.0 - g
    else:
        raise ValueError(f"unknown gripper convention {convention!r}")
    return np.clip(c, 0.0, 1.0)


@dataclass
class GripperMap:
    kind: str                       # 'tendon_lut' | 'linear'
    xml_path: str
    joint_names: Sequence[str]      # mechanism joints, chain order
    qpos_idx: np.ndarray            # their indices in the full qpos / chain pose
    n_arm: int                      # number of arm joints preceding them
    gs: np.ndarray                  # (n,) closure samples
    lut: np.ndarray                 # (n, n_grip) resting qpos per closure sample
    lower: np.ndarray               # (n_grip,) joint lower limits
    upper: np.ndarray               # (n_grip,) joint upper limits
    actuator: Optional[str] = None
    ctrlrange: Optional[Tuple[float, float]] = None

    @property
    def n_grip(self) -> int:
        return len(self.qpos_idx)


def _find_finger_joints(model: mujoco.MjModel) -> Tuple[list, np.ndarray]:
    names = [model.joint(j).name for j in range(model.njnt)]
    idx = [j for j, n in enumerate(names) if FINGER_JOINT_RE.search(n)]
    if not idx:
        raise RuntimeError(f"no finger joints matched {FINGER_JOINT_RE.pattern} in {names}")
    # qpos addresses: every joint here is hinge/slide (1 dof); assert the block trails the arm joints
    qadr = np.array([model.jnt_qposadr[j] for j in idx])
    n_arm = int(qadr.min())
    expect = np.arange(n_arm, model.nq)
    if not np.array_equal(np.sort(qadr), expect):
        raise RuntimeError(f"finger joints are not the trailing qpos block: qpos {sorted(qadr)} vs nq {model.nq}")
    return [names[j] for j in idx], qadr


def _find_actuator(model: mujoco.MjModel) -> Optional[int]:
    for cand in ACTUATOR_CANDIDATES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, cand)
        if aid >= 0:
            return aid
    return None


def _settle_lut(model, aid, qpos_idx, n_arm, n_samples, settle_steps) -> Tuple[np.ndarray, np.ndarray]:
    """Command the tendon actuator at n_samples closures, hold the arm at 0, step, record resting finger qpos.

    Only the arm velocity is zeroed (the arm is held, the mechanism moves freely), so the fingers reach the
    closed limit within ``settle_steps``."""
    data = mujoco.MjData(model)
    lo, hi = model.actuator_ctrlrange[aid]
    arm_idx = np.arange(n_arm)
    gs = np.linspace(0.0, 1.0, n_samples)
    lut = np.zeros((n_samples, len(qpos_idx)))
    for i, g in enumerate(gs):
        mujoco.mj_resetData(model, data)
        data.ctrl[aid] = lo + g * (hi - lo)
        for _ in range(settle_steps):
            data.qpos[arm_idx] = 0.0
            data.qvel[arm_idx] = 0.0
            mujoco.mj_step(model, data)
        lut[i] = data.qpos[qpos_idx]
    return gs, lut


def build_gripper_map(model_xml_dir: str, n_samples: int = 33, settle_steps: int = 400) -> GripperMap:
    """Build the mapping for a render model from its ``robot_xml`` directory (or MJCF file)."""
    xml_path = resolve_model_xml(model_xml_dir)
    model = mujoco.MjModel.from_xml_path(xml_path)
    joint_names, qpos_idx = _find_finger_joints(model)
    n_arm = int(qpos_idx.min())
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    lower = model.jnt_range[jids, 0].copy()
    upper = model.jnt_range[jids, 1].copy()
    aid = _find_actuator(model)
    act_name = model.actuator(aid).name if aid is not None else None
    ctrlrange = tuple(float(x) for x in model.actuator_ctrlrange[aid]) if aid is not None else None

    is_slide = all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in jids)
    if aid is not None and model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_TENDON:
        gs, lut = _settle_lut(model, aid, qpos_idx, n_arm, n_samples, settle_steps)
        kind = "tendon_lut"
        travel = lut[-1, 0] - lut[0, 0]
        if travel < 0.5:
            raise RuntimeError(f"settled LUT barely moves the driver ({lut[0, 0]:.3f} -> {lut[-1, 0]:.3f} rad)")
    elif is_slide and len(jids) == 2:
        # WidowX-style parallel jaw: actuator ctrlrange gives the open (upper) and closed (lower) finger positions;
        # the second finger mirrors the first (equality constraint polycoef -1).
        o, c = (ctrlrange[1], ctrlrange[0]) if ctrlrange else (upper[0], lower[0])
        gs = np.linspace(0.0, 1.0, n_samples)
        left = o - gs * (o - c)
        sign = np.sign(model.jnt_range[jids[1]].mean()) or -1.0
        lut = np.stack([left, sign * left], axis=1)
        kind = "linear"
    else:
        raise RuntimeError(f"unsupported gripper mechanism: joints {joint_names}, actuator {act_name}")
    return GripperMap(kind=kind, xml_path=xml_path, joint_names=joint_names, qpos_idx=qpos_idx, n_arm=n_arm,
                      gs=gs, lut=lut, lower=lower, upper=upper, actuator=act_name, ctrlrange=ctrlrange)
