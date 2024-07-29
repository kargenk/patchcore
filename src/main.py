from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from model import PatchCore

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'mvtec_ad', 'cable')
OUTPUT_DIR = ROOT_DIR.joinpath('outputs', 'mvtec_ad', 'cable')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchCore(device=device, file_name='mvtec-ad_cable.pickle', out_dir=OUTPUT_DIR)
    train_set = MVTecDataset(data_dir=DATA_DIR, phase='train/good')
    train_loader = DataLoader(train_set, batch_size=8)
    val_set = MVTecDataset(data_dir=DATA_DIR, phase='test/bent_wire')
    val_loader = DataLoader(val_set, batch_size=1)

    # # 正常画像空間の特徴を作成
    # for i, batch in tqdm(enumerate(train_loader)):
    #     model.add_memory(batch.to(device))
    # model.save_memories()

    for i, batch in tqdm(enumerate(val_loader)):
        model.inference(batch.to(device), i)
