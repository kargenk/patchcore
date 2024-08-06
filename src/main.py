from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset, unfold_collate
from model import PatchCore

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'kintsugi', 'image')
OUTPUT_DIR = ROOT_DIR.joinpath('outputs', 'kintsugi', 'image')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchCore(
        threshold=14,
        device=device,
        file_name='kintsugi_normal_res256_unfold_coreset_1percent.pickle',
        out_dir=OUTPUT_DIR,
        unfold=False
    )  # fmt: skip
    train_set = MVTecDataset(data_dir=DATA_DIR, phase='normal', size=256)
    train_loader = DataLoader(train_set, batch_size=8)
    val_set = MVTecDataset(data_dir=DATA_DIR, phase='anomaly', size=256)
    val_loader = DataLoader(val_set, batch_size=1)

    # 正常画像空間の特徴を作成
    for i, batch in tqdm(enumerate(train_loader)):
        imgs, imgs_path = batch
        model.add_memory(imgs.to(device))
    model.save_memories(save_ratio=0.01)
    print(f'Save Memorybank.')

    for i, batch in tqdm(enumerate(val_loader)):
        imgs, imgs_path = batch
        model.inference(imgs.to(device), Path(imgs_path[0]).stem)
