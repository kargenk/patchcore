from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


def inv_transform(x: torch.Tensor, inv_mean: list[float], inv_std: list[float]) -> torch.Tensor:
    """正規化した画像テンソルをもとに戻す

    Args:
        x (torch.Tensor): 正規化された画像テンソル
        inv_mean (List[float]): ImageNetの場合、[-0.485/0.229, -0.456/0.224, -0.406/0.255]
        inv_std (List[float]): ImageNetの場合、[1/0.229, 1/0.224, 1/0.255]

    Returns:
        torch.Tensor: [0.0, 1.0]から[0, 255]に戻した画像
    """
    normalize = transforms.Normalize(mean=inv_mean, std=inv_std)
    return normalize(x)


def restore_image(img_tensor: torch.Tensor, show: bool = False) -> np.ndarray:
    """正規化の逆変換を行い，元の画像に復元

    Args:
        img_tensor (torch.Tensor): 前処理後の画像[C, H, W].
        show (bool, optional): 表示するか. Defaults to False.

    Returns:
        np.ndarray: 復元した画像
    """
    imagenet_inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255]
    imagenet_inv_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.255]

    img_tensor = inv_transform(img_tensor, imagenet_inv_mean, imagenet_inv_std)
    img_arr = img_tensor.permute(1, 2, 0).cpu().numpy()

    if show:
        import matplotlib.pyplot as plt

        plt.imshow(img_arr)
        plt.show()

    return img_arr


def cvt2heatmap(gray: np.ndarray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap: np.ndarray, image: np.ndarray):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def save_anomaly_map(save_dir, anomaly_map, input_img, file_name, norm_max=3):
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
    print(str(save_dir / f'{file_name}.png'))
    cv2.imwrite(str(save_dir / f'{file_name}.png'), input_img)
    cv2.imwrite(str(save_dir / f'{file_name}_amap.png'), anomaly_map_norm_hm)
    cv2.imwrite(str(save_dir / f'{file_name}_amap_on_img.png'), hm_on_img)
