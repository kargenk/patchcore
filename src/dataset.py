from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from plot_utils import restore_image

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'mvtec_ad', 'bottle')


class MVTecDataset(Dataset):
    def __init__(self, data_dir: Path, phase: str = 'train', size: int = 256, unfold: bool = False):
        super().__init__()
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # PILが大きなイメージをロードしない仕様のため
        self.data_dir = data_dir
        self.phase = phase
        self.size = size
        self.unfold = unfold

        self.img_list = list(self.data_dir.joinpath(self.phase).glob(f'*.png'))
        self.img_size = Image.open(self.img_list[0]).convert('RGB').size

        if self.unfold:
            # 入力画像をパッチできれいに分けられるサイズに拡大するための拡大係数
            scale_factor = int(max(self.img_size) / self.size) + 1
            up_size = self.size * scale_factor

            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(up_size, up_size)),
                    transforms.PILToTensor(),
                    Unfold(size=256),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(self.size, self.size)),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.img_list[idx]
        img_pil = Image.open(img_path).convert('RGB')
        return self.transforms(img_pil), str(img_path)

    def __len__(self):
        return len(self.img_list)

class Unfold:
    """入力画像をパッチ化する変換.Input Size=[N, C, H, W]
    """
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)  # add batch dim
        # [N, C, H, W] -> [N, C, H//kernel, W//kernel, kernel, kernel]
        patches = x.unfold(2, self.size, self.size).unfold(3, self.size, self.size)
        # [N, C, H//kernel, W//kernel, kernel, kernel] -> [N * H//kernel * W//kernel, C, kernel, kernel]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, self.size, self.size)
        return patches

def _default_collate_fn(batch: list[tuple[torch.Tensor], tuple[str]]) -> tuple[torch.Tensor, tuple[str]]:
    """デフォルトのcollate_fn(バッチ化関数).

    Args:
        batch (list[tuple[torch.Tensor], tuple[str]]): 画像テンソルのタプルとファイルパス一覧のタプル

    Returns:
        tuple[torch.Tensor, tuple[str]]: バッチ化した画像とファイルパス
    """
    images, imgs_path = list(zip(*batch))
    images = torch.stack(images)
    return images, imgs_path

def unfold_collate(batch: list[tuple[torch.Tensor], tuple[str]]) -> tuple[torch.Tensor, tuple[str]]:
    """入力画像をunfoldする場合のcollate_fn(バッチ化関数).

    Args:
        batch (list[tuple[torch.Tensor], tuple[str]]): 画像テンソルのタプルとファイルパス一覧のタプル

    Returns:
        tuple[torch.Tensor, tuple[str]]: バッチ化した画像とファイルパス
    """
    images, imgs_path = list(zip(*batch))
    images = torch.cat(images, dim=0)
    return images, imgs_path

if __name__ == '__main__':
    dataset = MVTecDataset(data_dir=DATA_DIR, phase='train/good', size=256, unfold=False)
    img_tensor, img_path = iter(dataset).__next__()
    print(f'Dataset Size: {len(dataset)}')
    print(f'Data Shape: {img_tensor.shape}')
    print(img_path)

    # # 画像表示（デバッグ用）
    # img_arr = restore_image(img_tensor, show=False)

    # 入力画像が大きく、異常が小さい場合
    unfold_dataset = MVTecDataset(data_dir=DATA_DIR, phase='train/good', size=256, unfold=True)
    unfold_dataloader = DataLoader(unfold_dataset, batch_size=1, collate_fn=unfold_collate)
    unfold_img_tensor, unfold_img_path = unfold_dataloader.__iter__().__next__()
    print(f'Batch Shape: {unfold_img_tensor.shape}')
