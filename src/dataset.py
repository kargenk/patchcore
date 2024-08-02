from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from plot_utils import restore_image

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'mvtec_ad', 'bottle')


class MVTecDataset(Dataset):
    def __init__(self, data_dir: Path, phase: str = 'train', resolution: int = 256):
        super().__init__()
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # PILが大きなイメージをロードしない仕様のため
        self.data_dir = data_dir
        self.phase = phase
        self.resolution = resolution

        self.img_list = list(self.data_dir.joinpath(self.phase).glob(f'*.png'))

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(self.resolution, self.resolution)),
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


if __name__ == '__main__':
    dataset = MVTecDataset(data_dir=DATA_DIR, phase='train/good', resolution=256)
    img_tensor, img_path = iter(dataset).__next__()
    print(f'Dataset Size: {len(dataset)}')
    print(f'Data Shape: {img_tensor.shape}')
    print(img_path)

    # # 画像表示（デバッグ用）
    # img_arr = restore_image(img_tensor, show=False)

    # collate_fnによるpatchify
    my_dataloader = DataLoader(dataset, batch_size=1)
    my_img_tensor, my_img_path = my_dataloader.__iter__().__next__()
    print(f'Batch Shape: {my_img_tensor.shape}')
