from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from model import PatchCore

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'necp', 'back')
OUTPUT_DIR = ROOT_DIR.joinpath('outputs', 'necp', 'back')

if __name__ == '__main__':
    size = 1024
    save_ratio = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchCore(
        threshold=10,
        device=device,
        file_name=f'necp_back_res{size}_sub_coreset_{int(save_ratio * 100)}percent.pickle',
        out_dir=OUTPUT_DIR,
        unfold=False
    )  # fmt: skip
    train_set = MVTecDataset(data_dir=DATA_DIR, phase='ok/subset', size=size)
    train_loader = DataLoader(train_set, batch_size=8)
    val_set = MVTecDataset(data_dir=DATA_DIR, phase='ng', size=size)
    val_loader = DataLoader(val_set, batch_size=1)

    # 正常画像空間の特徴を作成
    for i, batch in tqdm(enumerate(train_loader)):
        imgs, imgs_path = batch
        model.add_memory(imgs.to(device))
    model.save_memories(save_ratio=save_ratio)
    print(f'Save Memorybank.')

    for i, batch in tqdm(enumerate(val_loader)):
        imgs, imgs_path = batch
        model.inference(imgs.to(device), Path(imgs_path[0]).stem)
