import torch
import numpy as np
from piq import DISTS, LPIPS
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.metric_util import to_y_channel

_lpips = None
_dists = None

def get_lpips() -> LPIPS:
    global _lpips
    if _lpips is None:
        _lpips = LPIPS()
        _lpips.eval()

    return _lpips

def get_dists() -> DISTS:
    global _dists
    if _dists is None:
        _dists = DISTS()
        _dists.eval()

    return _dists

def to_tensor(img: np.ndarray) -> torch.Tensor:
    return (torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255.0))

@METRIC_REGISTRY.register()
def calculate_lpips(img: np.ndarray, img2: np.ndarray, crop_border: int, input_order: str = 'HWC', test_y_channel: bool = False, **kwargs) -> float:

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ('HWC', 'CHW'):
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    if input_order == 'CHW':
        img  = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ten1 = to_tensor(img)
    ten2 = to_tensor(img2)

    device = next(get_lpips().parameters()).device
    ten1 = ten1.to(device)
    ten2 = ten2.to(device)

    with torch.no_grad():
        score = get_lpips()(ten1, ten2)

    return float(score.item())

@METRIC_REGISTRY.register()
def calculate_dists(img: np.ndarray, img2: np.ndarray, crop_border: int, input_order: str = 'HWC', test_y_channel: bool = False, **kwargs) -> float:

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ('HWC', 'CHW'):
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    if input_order == 'CHW':
        img  = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ten1 = to_tensor(img)
    ten2 = to_tensor(img2)

    device = next(get_dists().parameters()).device
    ten1 = ten1.to(device)
    ten2 = ten2.to(device)

    with torch.no_grad():
        score = get_dists()(ten1, ten2)

    return float(score.item())