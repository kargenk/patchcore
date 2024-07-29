from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'mvtec_ad', 'bottle')

class MVTecDataset(Dataset):
    def __init__(self, data_dir: Path, phase: str = 'train'):
        super().__init__()
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # PILが大きなイメージをロードしない仕様のため
        self.data_dir = data_dir
        self.phase = phase

        self.img_list = list(self.data_dir.joinpath(self.phase).glob(f'*.png'))

        self.transforms = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.img_list[idx]
        img_pil = Image.open(img_path).convert('RGB')
        return self.transforms(img_pil)

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    dataset = MVTecDataset(data_dir=DATA_DIR, phase='train/good')
    print(f'Dataset Size: {len(dataset)}')
    print(f'Data Shape: {iter(dataset).__next__().shape}')
