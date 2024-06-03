from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


def inv_transform(x: torch.Tensor, inv_mean: list[float], inv_std: list[float]) -> torch.Tensor:
    """正規化した画像テンソルをもとに戻す

    Args:
        x (torch.Tensor): 正規化された画像テンソル
        inv_mean (List[float]): [-0.485/0.229, -0.456/0.224, -0.406/0.255]
        inv_std (List[float]): [1/0.229, 1/0.224, 1/0.255]

    Returns:
        torch.Tensor: _description_
    """
    return transforms.Normalize(
        mean=inv_mean, std=inv_std
    )(x)

def cvt2heatmap(gray:np.ndarray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap: np.ndarray, image: np.ndarray):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)

def save_anomaly_map(save_dir, anomaly_map, input_img, gt_img, file_name, x_type, norm_max=3):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
    print(f'ano map min: {anomaly_map.min()} max: {anomaly_map.max()} in {file_name}')
    anomaly_map[anomaly_map > norm_max] = norm_max
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / norm_max
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_img)

    # save images
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(str(save_dir / f'{x_type}_{file_name}.png'))
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}.png'), input_img)
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}_amap.png'), anomaly_map_norm_hm)
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}_amap_on_img.png'), hm_on_img)
    cv2.imwrite(str(save_dir / f'{x_type}_{file_name}_gt.png'), gt_img)