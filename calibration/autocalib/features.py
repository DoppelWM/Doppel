"""DINOv2 patch features for the appearance loss."""
import torch
import torch.nn.functional as F
import torchvision.transforms as T


_DINO_CACHE = {}


def cosine_loss(a, b):
    """Mean of 1 - cosine similarity along dim 0 (the batch axis for (B, N, D) tokens)."""
    assert a.shape == b.shape
    return (1 - F.cosine_similarity(a, b, dim=0)).mean()


def load_dinov2(name, device='cuda'):
    """torch.hub facebookresearch/dinov2 backbone (cached per process)."""
    if name not in _DINO_CACHE:
        _DINO_CACHE[name] = torch.hub.load('facebookresearch/dinov2', name).to(device)
    return _DINO_CACHE[name]


def resize_tensor(image, longest_edge=800):
    """Bilinear-resize a (B, C, H, W) tensor so that its longest edge equals `longest_edge`."""
    B, C, H, W = image.shape
    ratio = longest_edge / W if W > H else longest_edge / H
    new_W = int(W * ratio)
    new_H = int(H * ratio)
    return F.interpolate(image, size=(new_H, new_W), mode='bilinear', align_corners=False)


def interpolate_to_patch_size(img_bchw, patch_size):
    """Nearest-resize so H and W are multiples of the ViT patch size."""
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W


NORMALIZE = T.Normalize(mean=[0.5], std=[0.5])


def patch_tokens(dinov2, images_bchw01, longest_edge=800):
    """DINOv2 `x_norm_patchtokens` for a batch of [0,1] images; returns (tokens (B, N, D), grid_h, grid_w)."""
    x = NORMALIZE(resize_tensor(images_bchw01, longest_edge=longest_edge))
    x, target_H, target_W = interpolate_to_patch_size(x, dinov2.patch_size)
    tokens = dinov2.forward_features(x)["x_norm_patchtokens"]
    return tokens, target_H // dinov2.patch_size, target_W // dinov2.patch_size
