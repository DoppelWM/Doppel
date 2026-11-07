"""Per-video top-camera calibration for the bimanual YAM stations of ABC-130K.

Pipeline (``run_videos.py``): decode ~12-20 frames and the joint states of one episode (``extract.py``),
segment the two arms with SAM 2 prompted from a coarse robot render (``masks.py``), fit the camera pose to the
masks (soft IoU, ``fit.py``), refine it photometrically, and gate the result (``export.py``).
"""
