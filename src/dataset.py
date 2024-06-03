from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data')

class CosmeDataset(Dataset):
    def __init__(self, data_dir: Path, phase: str = 'train'):
        super().__init__()
        self.data_dir = data_dir
        if phase == 'train':
            ext = '.jpg'
        elif phase == 'val':
            ext = '.png'
        self.phase = phase
        self.img_list = list(self.data_dir.joinpath(self.phase).glob(f'*{ext}'))

        self.transforms = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.img_list[idx]
        img_pil = Image.open(img_path).convert('RGB')
        return self.transforms(img_pil), torch.randn(1), torch.randn(1), torch.randn(1)

    def __len__(self):
        return len(self.img_list)


class MVTecDataset(Dataset):
    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    dataset = CosmeDataset(data_dir=DATA_DIR, phase='train')
    print(iter(dataset).__next__().shape)
