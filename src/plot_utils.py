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


def make_heatmap(gray: np.ndarray) -> np.ndarray:
    """グレースケール画像からヒートマップを作成.

    Args:
        gray (np.ndarray): グレースケール画像のNumPy配列

    Returns:
        np.ndarray: ヒートマップ画像
    """
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
    """ヒートマップを入力画像に重ね合わせて返す.

    Args:
        heatmap (np.ndarray): ヒートマップ画像
        image (np.ndarray): 元画像

    Returns:
        np.ndarray: ヒートマップを重ね合わせた画像
    """
    res = np.float32(heatmap) / 255 + np.float32(image) / 255
    res = res / np.max(res)
    return np.uint8(255 * res)


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """グレースケール画像をmin-max正規化して返す.

    Args:
        image (np.ndarray): 画像のNumPy配列Shape=[H, W], [-∞, ∞]

    Returns:
        np.ndarray: 正規化した画像[0, 1]
    """
    _min, _max = image.min(), image.max()
    normalized = (image - _min) / (_max - _min)
    return normalized


def save_anomaly_map(save_dir: Path, anomaly_map: np.ndarray, input_img: np.ndarray, file_name: str) -> None:
    """オリジナル/ヒートマップ/重ね合わせ画像を保存

    Args:
        save_dir (Path): 保存先のディレクトリ
        anomaly_map (np.ndarray): 異常箇所のセグメンテーション画像(グレースケール)
        input_img (np.ndarray): 入力画像(RGB)
        file_name (str): 保存ファイル名
    """
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    # 異常箇所の正規化
    anomaly_map_norm = min_max_normalize(anomaly_map)
    # print(f'ano map min: {anomaly_map.min()} max: {anomaly_map.max()} in {file_name}')

    # 異常箇所をヒートマップとして元画像に重ね合わせる
    anomaly_heatmap = make_heatmap(anomaly_map_norm * 255)
    hm_on_img = overlay_heatmap(anomaly_heatmap, input_img)

    # オリジナル/ヒートマップ/重ね合わせ画像を保存
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(str(save_dir / f'{file_name}.png'))
    cv2.imwrite(str(save_dir / f'{file_name}.png'), input_img)
    cv2.imwrite(str(save_dir / f'{file_name}_amap.png'), anomaly_heatmap)
    cv2.imwrite(str(save_dir / f'{file_name}_amap_on_img.png'), hm_on_img)
