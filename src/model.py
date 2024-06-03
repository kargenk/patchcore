import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor

from plot_utils import inv_transform, save_anomaly_map
from sampler import KCenterGreedy

ROOT_DIR = Path(__file__).parents[1]

class PatchCore(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.model = resnet50(pretrained=True)
        self.extractor = create_feature_extractor(self.model, {'layer2': 'mid1', 'layer3': 'mid2'})
        self.memory_bank = []
        self.save_dir = ROOT_DIR.joinpath('outputs')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴抽出を行う

        Args:
            x (torch.Tensor): 入力

        Returns:
            dict[str, torch.Tensor]: 中間特徴量の辞書
        """
        self.feature = self.extractor(x)  # layer2: [N, 512, H/8, W/8], layer3: [N, 1024, H/16, W/16]
        return self.feature

    def reshape_embedding(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """N x H x W 個のチャネル方向特徴量のリストを作成.

        Args:
            embedding (torch.Tensor): torch.Size([N, 1536, H/8, W/8])

        Returns:
            list[torch.Tensor]: torch.Size([1536], ...)
        """
        for i in range(embedding.shape[0]):
            for j in range(embedding.shape[2]):
                for k in range(embedding.shape[3]):
                    self.memory_bank.append(embedding[i, :, j, k])
        return self.memory_bank

    def emb_concat(self, embeddings:list[torch.Tensor]) -> torch.Tensor:
        """2つの中間特徴量のうち、小さい方をバイリニア補間でアップサンプリングしてからチャネル方向に結合.

        Args:
            embeddings (dict[str, torch.Tensor]): 浅くも深くもない位置の中間特徴量二つ

        Returns:
            torch.Tensor: チャネル方向に結合した中間特徴量
        """
        B, C1, H1, W1 = embeddings[0].size()  # torch.Size([N, 512, H/8, W/8])
        _, C2, H2, W2 = embeddings[1].size()  # torch.Size([N, 1024, H/16, W/16])
        s = int(H1 / H2)  # 2
        mid2 = F.interpolate(embeddings[1], scale_factor=s, mode='bilinear')
        return torch.cat([embeddings[0], mid2], dim=1)

    def add_memory(self, batch) -> None:
        """位置毎の特徴量をMemoryBankに格納(バッチごとに処理を行う)

        Args:
            batch (object): 入力のバッチ

        Returns:
            _type_: _description_
        """
        x, _, _, _ = batch
        features = self.extractor(x)

        # 位置に敏感にならないようにAdaptive Poolingで周囲と混ぜる（ぼかす）
        embeddings = []
        for _, feature in features.items():
            m = nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature.cpu()))
        # 特徴マップを結合
        embedding = self.emb_concat(embeddings)
        # 位置毎の特徴量を格納
        embedding = self.reshape_embedding(embedding.detach().numpy())
        self.memory_bank.extend(embedding)

    def save_memories(self) -> None:
        """高速化のため、Random Projectionを使用して次元削減後、Greedy法を用いて特徴量のサンプリングを行って保存する.
        """
        total_embeddings = np.array(self.memory_bank)
        # 高速化のため、Random Projectionで次元削減, 'auto' => Johnson-Lindenstrauss lemma
        # 高次元のユークリッド空間内の要素をそれぞれの要素間の距離をある程度保ったまま、
        # 別の(低次元の)ユークリッド空間へ線型写像で移せるという補題
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.randomprojector.fit(total_embeddings)

        # Greedy法を用いて、特徴量の数をN個選択する
        selector = KCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0] * 0.001))
        self.embedding_coreset = total_embeddings[selected_idx]
        print('initial embedding size : ', total_embeddings.shape)      # (245760, 1536)
        print('final embedding size : ', self.embedding_coreset.shape)  # (245, 1536)

        # 特徴量の保存
        with self.save_dir.joinpath('embedding.pickle').open('wb') as f:
            pickle.dump(self.embedding_coreset, f)

    def inference(self, batch, batch_idx):
        x, gt, label, x_type = batch
        # 正常画像の特徴量を読み込む
        with self.save_dir.joinpath('embedding.pickle').open('rb') as f:
            self.embedding_coreset = pickle.load(f)
        # テスト画像の特徴量を計算
        features = self.extractor(x)
        embeddings = []
        for _, feature in features.items():
            m = nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature.cpu()))
        # 特徴マップを結合
        embedding = self.emb_concat(embeddings)
        # 位置毎の特徴量を格納
        embedding = self.reshape_embedding(embedding.detach().numpy())

        # k近傍法で最も近い特徴量をn_neighbors個探索する
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        score_patches, _ = nbrs.kneighbors(embedding)  # 正解特徴量との距離(特徴マップ32x32=1024, 近傍特徴量n_neighbors)
        anomaly_map = score_patches[:, 0].reshape((32, 32))

        # 画像レベルの異常度を計算
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        scores = w * max(score_patches[:, 0])

        # 異常マップをリサイズ
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (254, 254))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(scores)

        # 画像の保存
        x = inv_transform(x, self.cfg.imdata.inv_img_mean, self.cfg.imdata.inv_img_std)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        file_name = f'{batch_idx :05d}'
        save_anomaly_map(self.save_dir, anomaly_map_resized_blur, input_x, gt_np * 255, file_name, x_type[0])

if __name__ == '__main__':
    dummy = torch.randn((8, 3, 256, 256))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patchcore = PatchCore(device=device)
    print(patchcore.add_memory((dummy, None, None, None)))
    print(len(patchcore.memory_bank))
