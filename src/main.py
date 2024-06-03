from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CosmeDataset
from model import PatchCore

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchCore(device=device)
    train_set = CosmeDataset(data_dir=DATA_DIR, phase='train')
    train_loader = DataLoader(train_set, batch_size=8)
    val_set = CosmeDataset(data_dir=DATA_DIR, phase='val')
    val_loader = DataLoader(val_set, batch_size=1)

    # 正常画像空間の特徴を作成
    for i, batch in tqdm(enumerate(train_loader)):
        model.add_memory(batch)
    model.save_memories()

    for i, batch in tqdm(enumerate(val_loader)):
        imgs, gts, labels, im_types = batch
        imgs = imgs.to(device)
        y = model(imgs)
        print(len(y), list(y.values())[0].shape, list(y.values())[1].shape)
        print('-' * 10)
        embeddings = []
        for _, feature in y.items():
            m = torch.nn.AvgPool2d(3, 1, 1)
            feature = m(feature)
            print(feature.shape)
            embeddings.append(feature)
        embedding = model.emb_concat(embeddings)
        print(embedding.shape) # torch.Size([2, 1536, 32, 32])
        x = model.reshape_embedding(embedding.detach().numpy()) # 2048 (1536,)
        print(len(x), x[0].shape)
        break
