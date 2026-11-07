"""Load a trained Dr.Robot model directory for calibration."""
import torch

from drrobot.gaussian_model import load_gaussians


def load_robot_model(model_path, background='white', device='cuda', iteration=None):
    """Return (gaussians, background colour tensor). ``background``: 'white' (calibration) or 'black'."""
    gaussians = load_gaussians(model_path, iteration=iteration, device=device)
    bg = [1, 1, 1] if background == 'white' else [0, 0, 0]
    return gaussians, torch.tensor(bg, dtype=torch.float32, device=device)
