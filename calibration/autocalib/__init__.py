"""Camera calibration from robot videos with a differentiable Gaussian-splat robot model.

The robot is rendered at the logged joints and a rigid camera pose is optimised until the rendered robot's
2-D motion (Gaussian flow), colour and DINOv2 features match the video.

    config     recipes and constants
    core       CameraOptimizer, the optimisation loop
    datasets   Bridge and RoboMIND drivers
    flow       RAFT / CoTracker3 dense flow of the real video
    features   DINOv2 patch features
    intrinsics MoGe focal-length estimate (Bridge)
    model      loading a Dr.Robot model directory
"""
