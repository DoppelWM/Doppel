# Camera calibration

Code that recovers the camera extrinsics we release for Bridge, RoboMIND and ABC. The robot is rendered with a
differentiable Gaussian-splat model ([Dr.Robot](https://arxiv.org/abs/2410.13851)) at the logged joint states, and a
6-DoF camera pose is optimised until the rendering matches the recorded video. The robot model stays fixed; only the
camera moves.

- **Bridge (WidowX) and RoboMIND (UR5e)**: one pose per recording session. This is the procedure of
  [RobotArena∞](https://proceedings.iclr.cc/paper_files/paper/2026/file/2aa3da3c1463ee2cdaaee94be4f8ba3f-Paper-Conference.pdf)
  (loss = flow + RGB + DINOv2 features), with RAFT flow instead of CoTracker, a grid of start poses scored on the
  first frames, and several trajectories of the same session fitted jointly.
- **ABC (bimanual YAM)**: one pose per video, fitted to SAM 2 masks of the two arms and then refined with a
  photometric loss.

`autocalib/` is the session-level pipeline, `yam/` the per-video ABC pipeline, `drrobot/` the renderer they share,
`tools/` data preparation, sweeps, export and overlays. Run the commands below from this directory.

## Installation

Python 3.10 and a CUDA GPU. Install PyTorch for your CUDA version, then

```bash
pip install -r requirements.txt
```

gsplat compiles its kernels on first use (needs `nvcc` and `ninja`; set `PYTHONUTF8=1` if the build fails with a
`UnicodeDecodeError`). Model weights are downloaded on first use:
DINOv2 ViT-S/14 and ViT-B/14 and CoTracker3 (`torch.hub`), RAFT-large (torchvision), MoGe `Ruicheng/moge-vitl` and
SAM 2.1 `facebook/sam2.1-hiera-small` (Hugging Face).

## Robot models

The Dr.Robot checkpoints we used are not released yet. Any Dr.Robot model trained from the robot's MJCF works; put it
in a directory with this layout:

```
models/<name>/
  robot_xml/model.xml (or scene.xml) + meshes        MJCF that defines the kinematic chain
  point_cloud/pose_conditioned_iteration_<N>/
    point_cloud.ply                                  canonical Gaussians
    lrs_model.pth                                    skinning-weight MLP
  canonical_joint_positions.npy                      optional: joint values of the canonical pose
```

The highest iteration is loaded. We used:

| dataset | robot | joints rendered | iteration |
|---|---|---|---|
| Bridge | WidowX-250s (MuJoCo Menagerie `trossen_wx250s`) | 6 arm + 2 fingers | highest saved |
| RoboMIND | UR5e + Robotiq 2F-85 | 6 arm + 8 gripper | 80,000 |
| ABC | one YAM arm (I2RT), rendered twice | 6 arm + 2 fingers per arm | 80,000 |

## Bridge

```bash
# raw layout with joint states from the LeRobot mirror of BridgeData V2, MoGe focal per trajectory
python tools/extract_bridge_lerobot.py --raw_root data/bridge_raw --episodes <episode indices>
python tools/moge_intrinsics.py --scene_path <trajectory dir>
# group trajectories into sessions (kitchen x camera variation x date), one focal per session
python tools/bridge_sessions.py --sessions sessions.csv --raw_root data/bridge_raw --out data/bridge_sweep
python tools/sweep_flow.py --dataset bridge --out_root data/bridge_sweep --stage flow --backend raft --gpus 0 1
python tools/sweep_calib.py --dataset bridge --out_root data/bridge_sweep --model_path models/widowx --gpus 0 1
```

Each session needs a start pose `init_w2c.npy` (a pose from a nearby session, or one made with
`tools/align_gui.py`). A single trajectory can be calibrated with `calibrate.py --dataset bridge`.

## RoboMIND

```bash
python tools/fetch_robomind.py --out data/robomind/h5_raw --embodiment h5_ur_1rgb --all_tasks --n_episodes 20
# -> data/robomind/survey/h5/<task>__<episode>.hdf5, then sessions (date + am/pm) and camera moves within them
python tools/robomind_sessions.py --survey_dir data/robomind/survey
python tools/sweep_flow.py --survey_dir data/robomind/survey --out_root data/robomind/sweep --stage all \
    --backend raft --init_w2c init.npy --gpus 0 1 2 3
python tools/sweep_calib.py --dataset robomind --out_root data/robomind/sweep --model_path models/ur5e_2f85 --gpus 0 1 2 3
```

`fetch_robomind.py` writes `<task>/<episode>/data/trajectory.hdf5`; link those files into `survey/h5/` under the name
`<task>__<episode>.hdf5`. The camera is the RealSense on top (`camera_top`, 640x480).

## Checking the session poses

The loss and motion recall do not tell a wrong basin from the right one. Run the sweep a second time from a different
start pose and compare:

```bash
python tools/sweep_calib.py ... --tag seedB --init_w2c other_start.npy
python tools/export_extrinsics.py --dataset robomind --out_root data/robomind/sweep --out robomind_sessions.csv
python tools/overlay.py --dataset robomind --model_path models/ur5e_2f85 --scene <traj>/camera_top
```

`export_extrinsics.py` marks a session `ok` when the two runs agree within 5 degrees and 5 cm. `overlay.py` renders
the robot over the frames (real | overlay | render); the silhouette should sit on the arm in every frame.

## ABC (YAM, per video)

```bash
python -m yam.run_videos --queue videos.csv --out_root data/abc/calib --model models/yam
python -m yam.export --root data/abc/calib --out data/abc/cameras.parquet --meta videos.csv
```

`videos.csv` has columns `uuid,task,mcap` and optionally `duration_s,camera,date` (camera and date are needed for the
same-day check). Per video: up to 20 frames are decoded from the MCAP and undistorted with the published calibration;
SAM 2 segments the two arms, prompted from a coarse render; the pose is searched over about 11,600 candidates (a grid
around the simulator camera and look-at cameras above the arms) and fitted to the masks on 6 frames (soft IoU, coarse
to fine), with 6 more frames held out; the result is refined with a photometric loss. Several workers can share one
queue (`--claims`). `python -m yam.extract`, `yam.masks` and `yam.fit` run the steps separately. A video takes about
1.5-2 minutes on an A100.

The photometric pose is kept unless it moved more than 3 degrees or 5 cm from the mask pose or lowered the held-out
IoU by more than 0.05. A video passes if its held-out mask IoU is at least 0.45, the camera is 1.3-2.2 m high and
20-75 degrees below horizontal, and, when the camera has at least 3 videos that day, the pose is within 5 cm and
3 degrees of that day's median. The YAM model directory must contain `canonical_joint_positions.npy`.

## Output format

Same conventions as the released extrinsics tables:

- `T_base_cam`: 4x4 camera-to-base transform, row-major, `x_base = T_base_cam @ x_cam`, OpenCV camera axes
  (x right, y down, z forward), metres. It is the inverse of the world-to-camera matrix the fit writes
  (`extrinsics.npy` for Bridge/RoboMIND, `w2c.npy` for ABC).
- Bridge and RoboMIND: robot base frame. ABC: the YAM station frame (left arm base at (0.2525, 0.31, 0.75), right
  arm base at y = 0.31 - 0.61, z up).
- `fx fy cx cy width height`: pinhole intrinsics of the frames the pose was fitted on, no distortion. The
  Bridge/RoboMIND renderer puts the principal point at the image centre, and the export reports it that way.

## Tests

```bash
pytest tests        # CPU only
```

## License

The files in `drrobot/` that carry the Inria header (`gaussian_model.py`, `renderer.py`, `cameras.py`,
`graphics.py`) are derived from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
are under the Gaussian-Splatting License (`LICENSE_GAUSSIAN_SPLATTING.md`, non-commercial research and evaluation
use only); `drrobot/pt3d.py` is from PyTorch3D (BSD-3-Clause). The rest of `drrobot/` (skinning, kinematics) follows
[Dr.Robot](https://github.com/cvlab-columbia/drrobot). Everything else is MIT, like the rest of this repository.

Models fetched at runtime have their own licenses: DINOv2 (Apache-2.0), RAFT via torchvision (BSD-3-Clause),
MoGe (MIT), SAM 2 (Apache-2.0), CoTracker3 (CC BY-NC 4.0, only used when a flow cache is missing and RAFT was not
run). The datasets keep their licenses: BridgeData V2 (CC BY 4.0), RoboMIND (gated, see its terms), ABC-130K (see
its terms).

## Citation

```bibtex
@inproceedings{jangir2026robotarena,
  title     = {{RobotArena} {$\infty$}: Scalable Robot Benchmarking via Real-to-Sim Translation},
  author    = {Jangir, Yash and Zhang, Yidi and Yamazaki, Kashu and Zhang, Chenyu and Tu, Kuan-Hsun and Ke, Tsung-Wei and
               Ke, Lei and Bisk, Yonatan and Fragkiadaki, Katerina},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}

@inproceedings{liu2024drrobot,
  title     = {Differentiable Robot Rendering},
  author    = {Liu, Ruoshi and Canberk, Alper and Song, Shuran and Vondrick, Carl},
  booktitle = {Conference on Robot Learning},
  year      = {2024}
}
```
